#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os, pickle, sys
from scoring import *
from utils import *
from tqdm import tqdm


"""
What this code needed !
1. sound category
2. which classifier you will be using
3. datadir, it should be in the format how your mudicove dir is
4. feats dir, where your all the feats will be. it should be in mudicov format
5. output dir, where you wanted to save your file

"""
#%%
sound_category=sys.argv[1]
classifier = sys.argv[2]
datadir=sys.argv[3]
featsfil = sys.argv[4]
outdir=sys.argv[5]  

#%%
cRange = [i for i in range(-7,8)]
#cRange = [i for i in range(-7,-6)]
categories = to_dict(datadir+"/category_to_class")
nfolds = open(datadir+'/nfolds').readlines()
nfolds = int(nfolds[0].strip())

if not os.path.exists(outdir):
	os.mkdir(outdir)


if classifier == 'lr':
    from sklearn.linear_model import LogisticRegression as clf
elif classifier in ['linearSVM','rbfSVM']:
    from sklearn.svm import SVC as clf
else:
    raise ValueError("Unknown classifier")


features_data=pd.read_csv(featsfil)

feat_list = list(features_data.columns.drop(["Unnamed: 0", "file_name"]))


## group the columns
feat_names = {}
for i in tqdm(range(0, len(feat_list) - 1)):
    feat_name_i = "_".join(feat_list[i].split("_")[0:-1])
    feat_name_ip1 = "_".join(feat_list[i + 1].split("_")[0:-1])
    if feat_name_i == feat_name_ip1:
        if feat_name_i not in feat_names.keys():
            feat_names[feat_name_i] = []
        feat_names[feat_name_i].append(feat_list[i].split("_")[-1]) 


feat_accuracy_dict = {}


#%%


for spec_feats in feat_names.keys():
    
    print(spec_feats)
    feat_ = [spec_feats+"_"+compo for compo in feat_names[spec_feats]]
    test_labels = to_dict(datadir+"/test_labels")

    for item in test_labels: test_labels[item]=categories[test_labels[item]]
    test_features_data = features_data[ features_data.file_name.isin(list(test_labels.keys()))]
    
    temp_for_index = test_features_data
    test_features_data = test_features_data[feat_]
    
    test_features_data['file_name'] = list(temp_for_index['file_name'].values)
    test_features_data = test_features_data.to_numpy()
    test_features = {}
    
    for item in test_features_data:
        test_features[item[-1]]=item[1:-1]
    del test_features_data,temp_for_index

    #bp()

    averageValidationAUCs={}
    print("=========== Tuning hyperparameters")
    for c in cRange:
        C = 10**c
     
        outfolder_root = "{}/results_{}_c{}".format(outdir,classifier,c)
        if not os.path.exists(outfolder_root):
            os.mkdir(outfolder_root)

        
        valAUCs=[]
        for fold in range(nfolds):

            train_labels = to_dict(datadir+'/fold_'+str(fold+1)+'/train_labels')
            for item in train_labels: train_labels[item]=categories[train_labels[item]]
            
            train_features_data = features_data[ features_data.file_name.isin(list(train_labels.keys()))]
            temp_for_index = train_features_data
            train_features_data = train_features_data[feat_]
            train_features_data['file_name'] = list(temp_for_index['file_name'].values)
            train_features_data = train_features_data.to_numpy()
            train_features = {}
            for item in train_features_data:
                train_features[item[-1]]=item[1:-1]
            del train_features_data, temp_for_index
            
            train_F=[]
            train_l=[]
            for item in train_labels:
                train_l.append(train_labels[item])
                train_F.append(train_features[item])
            train_F=np.array(train_F)
            train_l=np.array(train_l)
            
            
            scaler = StandardScaler()
            scaler.fit(train_F)
            if classifier=='lr':
                model = clf( C=C,
                            penalty='l2',
                            class_weight='balanced',
                            solver='liblinear',
                            random_state=np.random.RandomState(42))
            elif classifier=='linearSVM':
                model = clf( C=C,
                            kernel='linear',
                            class_weight='balanced',
                            probability=True,
                            random_state=np.random.RandomState(42))
            elif classifier=='rbfSVM':
                model = clf( C=C,
                            kernel='rbf',
                            class_weight='balanced',
                            probability=True,
                            random_state=np.random.RandomState(42))

            model.fit(scaler.transform(train_F),train_l)

            if not os.path.exists(outfolder_root+'/fold_'+str(fold+1)):
                os.mkdir(outfolder_root+'/fold_'+str(fold+1))

            pickle.dump({'classifier':model,'scaler':scaler},
                        open(outfolder_root+'/fold_'+str(fold+1)+'/model_'+spec_feats+'.pkl','wb'))

            #%%    
            val_labels = to_dict(datadir+'/fold_'+str(fold+1)+'/val_labels')
            for item in val_labels: val_labels[item]=categories[val_labels[item]]    
    
            val_features_data = features_data[ features_data.file_name.isin(list(val_labels.keys()))]
            temp_for_index = val_features_data
            val_features_data = val_features_data[feat_]
            val_features_data['file_name'] = list(temp_for_index['file_name'].values)
            val_features_data = val_features_data.to_numpy()
            val_features = {}
            for item in val_features_data:
                val_features[item[-1]]=item[1:-1]
            del val_features_data, temp_for_index

            

            scores={}
            for item in val_labels:
                feature = val_features[item]
                feature=feature.reshape(1,-1)
                scores[item] = model.predict_proba(scaler.transform(feature))[0][1]
        

            with open(outfolder_root+'/fold_'+str(fold+1)+'/val_scores_'+spec_feats+'.txt','w') as f:
                for item in scores: f.write("{} {}\n".format(item,scores[item]))
    
            scores = scoring(datadir+'/fold_'+str(fold+1)+'/val_labels',
                            outfolder_root+'/fold_'+str(fold+1)+'/val_scores_'+spec_feats+'.txt',
                            outfolder_root+'/fold_'+str(fold+1)+'/val_results_'+spec_feats+'.pkl')
            valAUCs.append(scores['AUC'])

            scores={}
            for item in test_labels:
                feature = test_features[item]
                feature=feature.reshape(1,-1)
                scores[item] = model.predict_proba(scaler.transform(feature))[0][1]

            with open(outfolder_root+'/fold_'+str(fold+1)+'/test_scores_'+spec_feats+'.txt','w') as f:
                for item in scores: f.write("{} {}\n".format(item,scores[item]))
    
            scores = scoring(datadir+'/test_labels',
                            outfolder_root+'/fold_'+str(fold+1)+'/test_scores_'+spec_feats+'.txt',
                            outfolder_root+'/fold_'+str(fold+1)+'/test_results_'+spec_feats+'.pkl')
            del scaler,model
        averageValidationAUCs[c]=sum(valAUCs)/len(valAUCs)

    best_c = max(averageValidationAUCs,key=averageValidationAUCs.get)
    bestC = 10**best_c
    print("Best Val. AUC {} for C={}".format(averageValidationAUCs[best_c],bestC))

    
    outfolder_root = "{}/results_{}".format(outdir,classifier)
    if not os.path.exists(outfolder_root):
        os.mkdir(outfolder_root)
    with open(outfolder_root+"/bestC","w") as f: 
        f.write("{}".format(bestC))
    
    train_labels = to_dict(datadir+"/train_labels")
    for item in train_labels: train_labels[item]=categories[train_labels[item]]

    train_features_data = features_data[ features_data.file_name.isin(list(train_labels.keys()))]

    temp_for_index = train_features_data
    train_features_data = train_features_data[feat_]
    train_features_data['file_name'] = list(temp_for_index['file_name'].values)
    train_features_data = train_features_data.to_numpy()
    train_features = {}

    for item in train_features_data: train_features[item[-1]]=item[1:-1]    

    train_F=[]
    train_l=[]
    for item in train_labels:
        train_l.append(train_labels[item])
        train_F.append(train_features[item])
    train_F=np.array(train_F)
    train_l=np.array(train_l)
    

    scaler = StandardScaler()
    scaler.fit(train_F)
    
    if classifier=='lr':
        model = clf( C=bestC,
                    penalty='l2',
                    class_weight='balanced',
                    solver='liblinear',
                    random_state=np.random.RandomState(42))
    elif classifier=='linearSVM':
        model = clf( C=bestC,
                    kernel='linear',
                    class_weight='balanced',
                    probability=True,
                    random_state=np.random.RandomState(42))
    elif classifier=='rbfSVM':
        model = clf( C=bestC,
                    kernel='rbf',
                    class_weight='balanced',
                    probability=True,
                    random_state=np.random.RandomState(42))

    model.fit(scaler.transform(train_F),train_l)

    pickle.dump({'classifier':model,'scaler':scaler},
                open(outfolder_root+'/model.pkl','wb'))

    #bp()
    scores={}
    for item in test_labels:
        feature = test_features[item]
        feature=feature.reshape(1,-1)
        scores[item] = model.predict_proba(scaler.transform(feature))[0][1]

    with open(outfolder_root+'/test_scores_'+spec_feats+'.txt','w') as f:
        for item in scores: f.write("{} {}\n".format(item,scores[item]))
    
    #bp()

    scores = scoring(datadir+'/test_labels',
                    outfolder_root+'/test_scores_'+spec_feats+'.txt',
                    outfolder_root+'/test_results_'+spec_feats+'.pkl')
    with open(outfolder_root+"/summary_"+spec_feats,"w") as f:
	    f.write("Test AUC {:.3f}\n".format(scores['AUC']))
	    for se,sp in zip(scores['sensitivity'],scores['specificity']):
		    f.write("Sensitivity: {:.3f}, Specificity: {:.3f}\n".format(se,sp))
    #bp()
    print("Test AUC: {}".format(scores['AUC']))
    feat_accuracy_dict[spec_feats] = scores['AUC']

sorted_acc = sorted(feat_accuracy_dict.values())
acc = list(feat_accuracy_dict.values())
key_list = list(feat_accuracy_dict.keys())


for val in sorted_acc:
    ind = acc.index(val)
    print(key_list[ind], val, file=open(sound_category+"_"+classifier+"_"+".txt", "a"))