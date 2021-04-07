# Evaluating future foehn development
This repository contains the code for the first part of my master's thesis where I evaluated future foehn development under a warming climate using machine learning. I experimented with different models - XGBoost, LightGBM, CatBoost, Random Forests, KNN with PCA preprocessing, DNNs, CNNs, regularized logistic regression. However, in the end, we decided to utilize XGBoost due to the best performance and less resource requirements than, say, CNNs. 

During the analysis, we train a model on reanalysis data (ERA-Interim), and generalize the model to a freely running climate simulation of present-day and future climate (CESM). Here, we optimize two objectives at the same time, the log loss and the squared mean loss. By that, we force the model into a trade-off: The former objective makes predictions as accurate on ERAI data (where we have a foehn label available), while the latter one aims to conserve the mean foehn frequency on CESM samples. Hence, we manage to reduce overfitting to ERAI samples and obtain valid foehn projections for climate simulations, which we test extensively.

# How to run
Since you will require the ERAI and CESM run data (which are property to the ETH IAC), just reach out to me and I will be happy to arrange something.

Afterward, install the conda environment & activate it:

```
conda env create -f environment.yml
conda activate foehn_development
```

Now you can run the code by yourself. The following section dives a little deeper into the skeleton of this project.

# Main structure 
The `data` folder contains parts of the initial data (since a climate simulation would be too large to copy) and all the data which will be generated during the analysis. However, this data is not included in the commits (due to file size).

The `figures` folder will contain all the figures which are generated as part of the analysis.

The scripts which are used for this analysis can be found under the `src` folder. There you can find the scripts which I used for the data reading & preprocessing under `01_preprocessing`. The folder `02_analysis` contains the scripts for various experiments, however the main script would be `src/02_analysis/train_model_on_ERAI_and_predict_on_CESM.ipynb`. Lastly, `03_plotting` contains a script for all kinds of plots which are not part of the main analysis, here plotting the ERAI and CESM topography for a given region of interest.

If you have any more questions, feel free to reach out to me:)

