# asp_project
COVID Audio Analysis - ASP Fall 21 Project

## Steps in running the code
1. Download the two folders `data` and `feats` from the link [here](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/rkumar44_jh_edu/Ek-JFg1syhBBto8XN5dG6N4BGQCiJ2Up2grXEUNx_o4cHg?e=wv1IPB).
2. Install the requirements to run the program using
```
 pip install -r requirements.txt 
 ```
3. extract the zip folders to this repo in your local machine
4. run the bash script `classifier.sh`
  ```
  bash classifier.sh
  ```
 ** Note - please change the bash file according to the alias used for running python scripts in your local machine. eg - replace python with python3 in the bash file if you have installed python with python3 alias.   
 
## Dataset
We used the CoSwara dataset for this project. The steps for downloading the dataset are listed below. We have the
```
git clone https://github.com/iiscleap/Coswara-Data.git
cd Coswara-Data
python extract_data.py
```
The feature extraction procedure is inspired from [MuDiCov](https://github.com/iiscleap/MuDiCov) Challenge Repo.

## About the feature extraction process.
For feature extraction we used an open source tool box called [Opensmile](https://www.audeering.com/research/opensmile/)

## References
[1] - Chetupalli, S.R., et al., Multi-modal Point-of-Care Diagnostics for COVID-19 Based On Acoustics and Symptoms. arXiv preprint arxiv:2106.00639, 2021.

[2] - Repository: [MuDiCov](https://github.com/iiscleap/MuDiCov)
