# asp_project
COVID Audio Analysis - ASP Fall 21 Project

## Steps in running the code
1. Download the two folders `data` and `feats` from the link [here](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/rkumar44_jh_edu/Ek-JFg1syhBBto8XN5dG6N4BGQCiJ2Up2grXEUNx_o4cHg?e=wv1IPB).
2. extract the zip folders to this repo in your local machine
3. run the bash script `classifier.sh`
  ``` 
  bash classifier.sh
  ```

## Create the meta file to be used for classification

```python
python create_meta_csv.py feats/modality_with_the_6k_features.csv output_meta.csv
```

The above command converts the feature csv file of 6k feat to the meta file required for classification




## Run classification on each feature of the ComPare2016 feature dataset

```python
python classification_individual_features.py breathing_meta.csv breathing_individual_features_classification_LR 0
```

The above command gives the accuracy corresponding to the features used for classification.
The above script can perform the following modes as per the required

"""
model - 0 - Linear Regression
model - 1 - SVM Linear
Model - 2 - Radial Basis Function SVM
"""
The C value in SVM play a major role
