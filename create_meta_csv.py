import pandas as pd
import sys
from tqdm import tqdm

feature_csv = sys.argv[1]
out_meta = sys.argv[2]


df = pd.read_csv(feature_csv)

meta = pd.read_csv('/Volumes/Datasets/Github/MuDiCov/data/metadata.csv')

labels_train = open("data/train_labels").readlines()
label_df = pd.DataFrame()
ids = []
label_list = []
for i in range(0, len(labels_train)):
    ids.append(labels_train[i][0:-3]) # 28 char of len
    if labels_train[i][-2:-1] == 'p':
        label_list.append(1) # c1 char of len
    elif labels_train[i][-2:-1] == 'n':
        label_list.append(0)

labels = open("data/test_labels").readlines()
for i in range(0, len(labels)):
    ids.append(labels[i][0:-3]) # 28 char of len
    if labels[i][-2:-1] == 'p':
        label_list.append(1) # c1 char of len
    elif labels[i][-2:-1] == 'n':
        label_list.append(0)
        
label_df['id'] = ids
label_df['target'] = label_list

meta_file = pd.DataFrame()
for i in tqdm(range(0, len(ids))):
    junk = (df.loc[df['file_name'] == ids[i]])
    junk = junk.drop(columns=["Unnamed: 0"])
    junk['target'] = label_list[i]
    meta_file = meta_file.append(junk, ignore_index = True)

meta_file.to_csv(out_meta)