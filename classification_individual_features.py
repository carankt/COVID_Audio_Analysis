from sklearn.linear_model import LogisticRegression as clf
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm
import numpy as np
import os, pickle, sys

meta_file = sys.argv[1]
out_file = sys.argv[2]
model_mode = sys.argv[3]
model_mode = int(model_mode)

""" 
model - 0 - Linear Regression
model - 1 - SVM Linear
Model - 2 - Radial Basis Function SVM 
"""

if model_mode == 0:
    from sklearn.linear_model import LogisticRegression as clf
    model_name = "LR"
elif model_mode == 1:
    from sklearn.svm import SVC as clf
    model_name = "SVM_Linear"
elif model_mode == 2:
    from sklearn.svm import SVC as clf
    model_name = "SVM_RBF"
else:
    raise ValueError("Unknown classifier")



ids_test = []
ids_train = []

labels = open("data/test_labels").readlines()
for i in range(0, len(labels)):
    ids_test.append(labels[i][0:-3]) # 28 char of len

labels = open("data/train_labels").readlines()
for i in range(0, len(labels)):
    ids_train.append(labels[i][0:-3]) # 28 char of len

speech_meta = pd.read_csv(meta_file)

feat_list = list(speech_meta.columns.drop(["Unnamed: 0", "file_name", "target"]))

## group the columns
feat_names = {}
for i in tqdm(range(0, len(feat_list) - 1)):
    feat_name_i = "_".join(feat_list[i].split("_")[0:-1])
    feat_name_ip1 = "_".join(feat_list[i + 1].split("_")[0:-1])
    if feat_name_i == feat_name_ip1:
        if feat_name_i not in feat_names.keys():
            feat_names[feat_name_i] = []
        feat_names[feat_name_i].append(feat_list[i].split("_")[-1])    
        
        
speech_meta_train = speech_meta[speech_meta["file_name"].isin(ids_train)]
speech_meta_test = speech_meta[speech_meta["file_name"].isin(ids_test)]
print(speech_meta_test.shape), print(speech_meta_train.shape)

feat_accuracy_dict = {}

## Model training and testing
for features in tqdm(feat_names.keys()):
    feat_ = []
    for val in feat_names[features]:
        feat_.append(features + "_" + val)
    print(features)
    scores = []
    input_to_model = speech_meta_train[feat_].to_numpy()
    output_of_the_model = speech_meta_train["target"].to_numpy()
    scaler = StandardScaler()
    scaler.fit(input_to_model)
    C = 10**2
    if model_mode == 0:
        model = clf( C=C,
                    penalty='l2',
                    class_weight='balanced',
                    solver='liblinear',
                    random_state=np.random.RandomState(42))
    elif model_mode == 1:
        model = clf( C=C,
                    kernel='linear',
                    class_weight='balanced',
                    probability=True,
                    random_state=np.random.RandomState(42))
    elif model_mode == 2:
        model = clf( C=C,
                    kernel='rbf',
                    class_weight='balanced',
                    probability=True,
                    random_state=np.random.RandomState(42))
        
    # training the model 
    model.fit(scaler.transform(input_to_model), output_of_the_model)
    
    # testing the model
    test_input_to_model = speech_meta_test[feat_].to_numpy()
    test_output_of_the_model = speech_meta_test["target"].to_numpy()
    
    output_predicted = model.predict(scaler.transform(test_input_to_model))
    accuracy = sum(output_predicted == test_output_of_the_model)/len(output_predicted)
    feat_accuracy_dict[features] = accuracy
    print(accuracy, features)
    
sorted_acc = sorted(feat_accuracy_dict.values())
acc = list(feat_accuracy_dict.values())
key_list = list(feat_accuracy_dict.keys())


for val in sorted_acc:
    ind = acc.index(val)
    print(key_list[ind], val, file=open(out_file + model_name + ".txt", "a"))
    
