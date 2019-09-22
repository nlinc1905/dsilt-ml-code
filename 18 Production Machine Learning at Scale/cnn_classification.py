import pandas as pd
import numpy as np
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
from keras.utils import np_utils
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix
from io import StringIO
import boto3


working_dir = "dsilt-ml-code/18 Production Machine Learning at Scale/aerial-cactus-identification/"
train_images_dir = working_dir+"train/train/"
test_images_dir = working_dir+"test/test/"

train = pd.read_csv(working_dir+"train.csv")
test = pd.read_csv(working_dir+"sample_submission.csv")
print(train.shape[0], "training images:", train['has_cactus'].value_counts()[1], 
      "images with cacti and", train['has_cactus'].value_counts()[0], "images without")
print(test.shape[0], "test images")


def img_to_array(img_file, width=32, height=32, channels=3):
    return np.array(img_file.getdata()).reshape(width, height, channels)

def load_images_from_dir(img_dir, file_extension='.jpg'):
    image_filenames = []
    images = []
    for filename in glob.glob(img_dir+'*'+file_extension):
        image_filenames.append(filename)
        with Image.open(filename) as im:
            images.append(img_to_array(im))
    return image_filenames, np.array(images)

x_train_fnames, x_train = load_images_from_dir(train_images_dir)
x_test_fnames, x_test = load_images_from_dir(test_images_dir)

# Normalize input values from 0-255 to 0-1
x_train = x_train/np.max([255, np.max(x_train)])
x_test = x_test/np.max([255, np.max(x_test)])
print("Sense check:\n",
      np.min(x_train), np.max(x_train), "\n",
      np.min(x_test), np.max(x_test))

# Sense check - make sure these look the same
plt.imshow(x_train[0])
plt.show()
im = Image.open(x_train_fnames[0])
im.show()

# Reorder data to match the order the files were read
train['id_reordered'] = pd.Categorical(
      train['id'],
      categories=[fname.split("\\")[1] for fname in x_train_fnames],
      ordered=True
)
train = train.sort_values('id_reordered')
train.reset_index(drop=True, inplace=True)
train.drop('id_reordered', axis=1, inplace=True)
test['id_reordered'] = pd.Categorical(
      test['id'],
      categories=[fname.split("\\")[1] for fname in x_test_fnames],
      ordered=True
)
test = test.sort_values('id_reordered')
test.reset_index(drop=True, inplace=True)
test.drop('id_reordered', axis=1, inplace=True)

# One hot encode outputs
y_train = np_utils.to_categorical(train['has_cactus'].values)
nbr_classes = y_train.shape[1]

# Create validation set
validation_split = 0.2
val_split_row = int(x_train.shape[0]*validation_split)
x_valid = x_train[:val_split_row,:]
x_train = x_train[val_split_row:,:]
y_valid = y_train[:val_split_row,:]
y_train = y_train[val_split_row:,:]
valid = train[:val_split_row].copy()
train = train[val_split_row:]


# Build the CNN
K.clear_session()
cnn_params = {'l1_activation': 'relu',
              'l2_activation': 'relu',
              'l3_activation': 'relu',
              'l4_activation': 'relu',
              'l5_activation': 'softmax',
              'optimizer': 'adam',
              'epochs': 5,      #Adjust to what works for the computer
              'batch_size': 200 #Adjust to what works for the computer
              }

model = Sequential()
model.add(Convolution2D(32, kernel_size=(3, 3),
                        strides=(1, 1),
                        padding='valid',
                        activation=cnn_params['l1_activation'],
                        bias_initializer='zeros',
                        kernel_initializer='glorot_uniform',
                        input_shape=(32, 32, 3),
                        data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, kernel_size=(3, 3),
                        strides=(1, 1),
                        padding='valid',
                        activation=cnn_params['l2_activation'],
                        bias_initializer='zeros',
                        kernel_initializer='glorot_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(128, kernel_size=(3, 3),
                        strides=(1, 1),
                        padding='valid',
                        activation=cnn_params['l3_activation'],
                        bias_initializer='zeros',
                        kernel_initializer='glorot_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512,
                bias_initializer='zeros',
                kernel_initializer='normal',
                activation=cnn_params['l4_activation']))
model.add(Dense(nbr_classes,
                bias_initializer='zeros',
                kernel_initializer='normal',
                activation=cnn_params['l5_activation']))
model.compile(loss='categorical_crossentropy',
              optimizer=cnn_params['optimizer'],
              metrics=['accuracy'])

cnn_model = model
print(cnn_model.summary())
print("Initial loss for softmax layer should be approximately:", \
      -np.log(1/nbr_classes))
cnn_model_hist = cnn_model.fit(x_train,
                               y_train,
                               validation_data=\
                               (x_valid,
                               y_valid),
                               epochs=\
                               cnn_params['epochs'],
                               batch_size=\
                               cnn_params['batch_size'],
                               verbose=2)
cnn_scores = cnn_model.evaluate(x_valid,
                                y_valid,
                                verbose=0)

print("Classification error: {}"\
      .format(round(100-cnn_scores[1]*100, 4)))
plt.plot(cnn_model_hist.history['acc'])
plt.plot(cnn_model_hist.history['val_acc'])
plt.title('CNN Model Training Curve')
plt.ylabel('Classification Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()
plt.plot(cnn_model_hist.history['loss'])
plt.plot(cnn_model_hist.history['val_loss'])
plt.title('CNN Model Training Curve')
plt.ylabel('Categorical Cross-Entropy Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()
# Evaluation on validation set
cnn_preds_prob = cnn_model.predict(x_valid)
cnn_preds = [np.argmax(x) for x in cnn_preds_prob]
true_labels = [np.argmax(x) for x in y_valid]
print("CNN Log Loss", log_loss(true_labels, cnn_preds_prob))
print("CNN Confusion Matrix\n", confusion_matrix(true_labels, cnn_preds))


# Submission to Kaggle
test_preds = cnn_model.predict_proba(x_test)[:, 1] # Only save prob of cactus (second column)
submission = pd.DataFrame({'id': test['id']})
submission['has_cactus'] = test_preds
#submission.to_csv(working_dir+"submission.csv", index=False)


def write_to_s3(data):
    csv_buffer = StringIO()
    data.to_csv(csv_buffer, index=False)
    try:
        s3 = boto3.resource(
            "s3",
            region_name="us-west-2",
            aws_access_key_id="YOUR_ACCESS_KEY_HERE",
            aws_secret_access_key="YOUR_SECRET_KEY_HERE",
        )
        s3.Object("YOUR_S3_BUCKETS_NAME_HERE", "submission.csv").put(
            Body=csv_buffer.getvalue()
        )
        return print("Write successful!")
    except:
        return print("Write failed.")

write_to_se(submission)
