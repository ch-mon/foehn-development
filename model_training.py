from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def fit_model_on_all_ERA_features_with_CV_and_return_most_important_features(model, parameters_CV, df_ERA, feature_names, train_mask, test_mask, with_CV=False):
    '''
    Train ML model on whole ERA data and return the most important features
    '''
    print("Started")
    
    if with_CV=True:    
        model_CV = GridSearchCV(model, parameters_CV, cv=3, n_jobs=5, scoring='neg_log_loss', verbose=3)
        model_CV.fit(df_ERA.loc[train_mask, feature_names], 
                     df_ERA.loc[train_mask, "Foehn"])
        
        print(model_CV.best_params_)
        best_model = model_CV.best_estimator_
    else:
        model.fit(df_ERA.loc[train_mask, feature_names], 
                 df_ERA.loc[train_mask, "Foehn"])
        best_model=model
    
    # Make predictions
    prediction_probas = best_model.predict_proba(df_ERA.loc[test_mask, feature_names])
    
    # Calculate best threshold
    precisions, recalls, thresholds = precision_recall_curve(df_ERA.loc[test_mask, "Foehn"], prediction_probas[:,1])
    best_threshold = thresholds[np.argmin(abs(precisions-recalls))-1]
    predictions = (prediction_probas[:,1]>best_threshold).astype(int)
    
    # Plot precision and recall curves
    f = plt.figure(figsize=(12,5))
    f.add_subplot(121)
    sns.lineplot(precisions, recalls)
    f.add_subplot(122)
    sns.lineplot(np.append(thresholds, 1.0), precisions)
    sns.lineplot(np.append(thresholds, 1.0), recalls)
    
    # Print best threshold, precision, recall and confusion matrix
    print(f"Best threshold: {best_threshold}")
    print(confusion_matrix(df_ERA.loc[test_mask, "Foehn"], predictions))
    print(f'Accuracy: {accuracy_score(df_ERA.loc[test_mask, "Foehn"], predictions)}')
    print(f'Precision: {precision_score(df_ERA.loc[test_mask, "Foehn"], predictions)}')
    print(f'Recall: {recall_score(df_ERA.loc[test_mask, "Foehn"], predictions)}')
    print(f'ROC-AUC: {roc_auc_score(df_ERA.loc[test_mask, "Foehn"], prediction_probas[:,1])}')
    print(f'Log-Loss: {log_loss(df_ERA.loc[test_mask, "Foehn"], prediction_probas[:,1])}')
    
    
    # Show 50 most important features
    df_ERA_feature_importances = pd.DataFrame({"feature_name": feature_names, "importance": best_model.feature_importances_}).sort_values(by="importance", ascending=False).reset_index(drop=True)
    display(df_ERA_feature_importances.head(50))
    
    return df_ERA_feature_importances

