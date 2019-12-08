# Name and matric number :
#   1. Siti Nur Shahirah Binti Wahab (192733)
#   2. Aishah Binti Ramli
# Description: Data Mining Assignment 2

#TODO: Import modules needed, declaration and csv file.
import pandas as pd
import numpy as np
import pickle as pck
import sklearn
from sklearn import preprocessing as pp
from sklearn.metrics import accuracy_score as ac
from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeClassifier as dtc

le = pp.LabelEncoder()

pd.set_option('display.max_columns', None)
df = pd.read_csv(r"adult.csv")

#TODO: Change and fill the missing values.
column_names = df.columns
for c in column_names:
    df[c] = df[c].replace("?", np.NaN)

df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))

#TODO: Discretisation of data in datasets (marital-status column)
df.replace(["Divorced", "Married-AF-spouse", "Married-civ-spouse", "Married-spouse-absent", "Never-married", "Separated", "Widowed"], \
            ["divorced", "married", "married", "married", "not married", "not married", "not married"], inplace = True)

#TODO: Label then encoder.
category_column =["workclass", "race", "education","marital-status", "occupation", "relationship", "gender", "native-country", "income"]
labelEncoder = pp.LabelEncoder()

#TODO: Create a map of all numerical values for each categorical labels.
map_dict={}
for column in category_column:
    df[column] = labelEncoder.fit_transform(df[column])
    le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
    map_dict[column]= le_name_mapping
print(map_dict)

#TODO: Drop all the redundant columns.
df=df.drop(["fnlwgt", "educational-num"], axis=1)

#TODO: Test the dataset accuracy using train_test_split() in sklearn.
X = df.values[:, 0:12]
Y = df.values[:, 12]

x_train, x_test, y_train, y_test = tts(X, Y, test_size = 0.3, random_state = 100)
dt_classify_gini = dtc(criterion = "gini", random_state = 100, max_depth=5, min_samples_leaf=5)
dt_classify_gini.fit(x_train, y_train)
y_predict_gini = dt_classify_gini.predict(x_test)

print ("\nDesicion Tree using Gini Index\nAccuracy is ", ac(y_test,y_predict_gini)*100 )

#TODO: Create and train a model, then serialize the model to a file named as prediction_model.pkl.
file = open(".../prediction_model.pkl", "wb")
pck.dump(dt_classify_gini, file)
