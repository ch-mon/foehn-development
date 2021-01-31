# Evaluating future foehn development
This repository contains the code for the first part of my master's thesis where I evaluated future foehn development under a warming climate using machine learning (actually I experimented with different models - XGBoost, LightGBM, CatBoost, Random Forests, KNN with PCA preprocessing, DNNs, CNNs, regularized logistic regression. However, in the end, we decided to utilize XGBoost due to the best performance and less resource requirements than, say, CNNs. 

Currently, I am in the process of cleaning the code, so that it is reproducible and shareable.

# How to run
Since you will require the ERAI and CESM run data (which are property to the ETH IAC), just reach out to me and I will be happy to arange something.

Afterward, install the conda environment & activate it:

```
conda env create -f environment.yml
conda activate foehn_development
```



# Main structure 
The scripts which are used for this analysis can be found under the `src` folder. There you can find the scripts which I used for the data preprocessing `01_preprocessing` and the analysis `02_analysis`. The main script is `src/02_analysis/train_model_on_ERAI_and_predict_on_CESM.ipynb`, which I have already started tidying up.

The `data` folder contains a structure (not the data itself) which I found helpful during the work and which serves as storage for intermediate data of different stages.

