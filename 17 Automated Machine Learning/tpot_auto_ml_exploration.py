from tpot import TPOTClassifier, TPOTRegressor
from sklearn.datasets import load_digits, load_boston
from sklearn.model_selection import train_test_split

export_dir = "dsilt-ml-code/17 Automated Machine Learning/"

#-------------------------------------------------------------------------------------------------#
#--------------------------------------Classification---------------------------------------------#
#-------------------------------------------------------------------------------------------------#

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)

tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export(export_dir+'tpot_mnist_pipeline.py')


#-------------------------------------------------------------------------------------------------#
#---------------------------------------Regression------------------------------------------------#
#-------------------------------------------------------------------------------------------------#

housing = load_boston()
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target,
                                                    train_size=0.75, test_size=0.25)

tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export(export_dir+'tpot_boston_pipeline.py')
