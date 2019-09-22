from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2
from PIL import Image
import pytesseract
import re
from textblob import Word


image = cv2.imread('dsilt-ml-code/13 Computer Vision/Tesseract OCR/street_sign.jpg')
orig = image.copy()
(H, W) = image.shape[:2]

# set parameters
confidence_thresh = 0.7

# EAST text requires that your input image dimensions be multiples of 32
def resize_image(im, h, w):
    largest_mult_less_than_h = h-(h % 32)
    largest_mult_less_than_w = w-(w % 32)
    global rW, rH
    rW = w / float(largest_mult_less_than_w)
    rH = h / float(largest_mult_less_than_h)
    return cv2.resize(im, (largest_mult_less_than_w, largest_mult_less_than_h))

image = resize_image(image, H, W)
(H, W) = image.shape[:2]

# define the two output layer names for the EAST detector model that we want
# output box probabilities and bounding box coordinates
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]

# load the pre-trained EAST text detector
net = cv2.dnn.readNet('dsilt-ml-code/13 Computer Vision/Tesseract OCR/frozen_east_text_detection.pb')

blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
    (123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []

for y in range(0, numRows):
    scoresData = scores[0, 0, y]
    xData0 = geometry[0, 0, y]
    xData1 = geometry[0, 1, y]
    xData2 = geometry[0, 2, y]
    xData3 = geometry[0, 3, y]
    anglesData = geometry[0, 4, y]

    for x in range(0, numCols):
        # if score does not have sufficient probability, ignore it
        if scoresData[x] < confidence_thresh:
            continue

        # compute the offset factor as the resulting feature maps will
        # be 4x smaller than the input image
        (offsetX, offsetY) = (x * 4.0, y * 4.0)

        # extract the rotation angle for the prediction and then
        # compute the sin and cosine
        angle = anglesData[x]
        cos = np.cos(angle)
        sin = np.sin(angle)

        # use the geometry volume to derive the width and height of
        # the bounding box
        h = xData0[x] + xData2[x]
        w = xData1[x] + xData3[x]
 
        # compute both the starting and ending (x, y)-coordinates for
        # the text prediction bounding box
        # add 10 pixels to add padding
        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))+10
        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))+15
        startX = int(endX - w)-20
        startY = int(endY - h)-20
        rects.append((startX, startY, endX, endY))
        confidences.append(scoresData[x])

# get rid of overlapping bounding boxes
boxes = non_max_suppression(np.array(rects), probs=confidences, overlapThresh=0.3)
orig_w_boxes = orig.copy()
 
for (startX, startY, endX, endY) in boxes:
    # scale the bounding box coordinates
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)
    cv2.rectangle(orig_w_boxes, (startX, startY), (endX, endY), (0, 255, 0), 2)

cv2.imshow("Text Detection", orig_w_boxes)
cv2.waitKey(0)

#------------------------------------------------------------------------------
#--------------------OCR Preprocessing (Image Cleaning)------------------------
#------------------------------------------------------------------------------

# Assume there is only 1 text bounding box (index 0)
image = orig[int(boxes[0][1]*rH):int(boxes[0][3]*rH), int(boxes[0][0]*rW):int(boxes[0][2]*rW)]
# Scale back to larger size
image_resized = cv2.resize(image, (W, int(H/3)))
gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, 
                                searchWindowSize=21)
#Denoising after Otsu's thresholding
otsu_thresh, otsu = cv2.threshold(gray, 0, 255, 
                                  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
otsu_den = cv2.fastNlMeansDenoising(otsu, h=10, templateWindowSize=7, 
                                    searchWindowSize=21)

cv2.imshow("Original", image)
cv2.imshow("Grayscale", gray)
cv2.imshow("Denoised", den)
cv2.imshow("Denoised OTSU", otsu)
cv2.imshow("OTSU Denoised", otsu_den)
cv2.waitKey(0)

#------------------------------------------------------------------------------
#--------------------------------OCR-------------------------------------------
#------------------------------------------------------------------------------

'''
#Page Segmentation (PSM) Settings:
  0    Orientation and script detection (OSD) only.
  1    Automatic page segmentation with OSD.
  2    Automatic page segmentation, but no OSD, or OCR.
  3    Fully automatic page segmentation, but no OSD. (Default)
  4    Assume a single column of text of variable sizes.
  5    Assume a single uniform block of vertically aligned text.
  6    Assume a single uniform block of text.
  7    Treat the image as a single text line.
  8    Treat the image as a single word.
  9    Treat the image as a single word in a circle.
 10    Treat the image as a single character.
 11    Sparse text. Find as much text as possible in no particular order.
 12    Sparse text with OSD.
 13    Raw line. Treat the image as a single text line,
            bypassing hacks that are Tesseract-specific.
#OCR Engine Mode (OEM) Settings:
  0    Legacy engine only.
  1    Neural nets LSTM engine only.
  2    Legacy + LSTM engines.
  3    Default, based on what is available.
'''

#Perform OCR
lg = 'eng'
conf = '--psm 3 --oem 2'
text_den = pytesseract.image_to_string(Image.fromarray(den), lang=lg, config=conf)
text_otsu_den = pytesseract.image_to_string(Image.fromarray(otsu_den), lang=lg, config=conf)

#Remove nonsense (extra whitespace, special chars that aren't punctuation)
pattern = re.compile(u"[^\w'\"!?.,]", re.UNICODE)

text_den = ' '.join(pattern.sub(' ', text_den).split())
text_otsu_den = ' '.join(pattern.sub(' ', text_otsu_den).split())
print(text_den)
print(text_otsu_den)

best = text_otsu_den
print(best)
with open("dsilt-ml-code/13 Computer Vision/Tesseract OCR/ocr_output.txt", "w") as text_file:
    text_file.write("Text Recognized: %s" % best)
