from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def fit_model_and_obtain_importances(model, df, feature_names, train_mask, test_mask, with_CV=False, parameters_CV=[]):
    """
    Train ML model on whole ERA data and return the most important features. Allows to train with or without CV.
    @param model: Any classification model that offers fit and predict_proba functions
    @param df: Dataframe with features and target ("Foehn")
    @param feature_names: Feature names
    @param train_mask: Mask for the training set
    @param test_mask: Mask for the test set
    @param with_CV: Whether to fit with CV
    @param parameters_CV: Which parameters to use for CV (dict)
    @return: Dataframe with sorted feature importances
    """

    print("Started fitting ...")
    
    if with_CV == True:  # Fit with cross-validation
        model_CV = GridSearchCV(model, parameters_CV, cv=5, n_jobs=5, scoring='neg_log_loss', verbose=3)
        model_CV.fit(df.loc[train_mask, feature_names],
                     df.loc[train_mask, "Foehn"])
        
        print(model_CV.best_params_)
        best_model = model_CV.best_estimator_
    else:  # Just fit a model normally
        model.fit(df.loc[train_mask, feature_names],
                  df.loc[train_mask, "Foehn"])
        best_model=model

    print("Finished fitting ...")
    
    # Make predictions
    prediction_probas = best_model.predict_proba(df.loc[:, feature_names])[:, 1]
    
    # Calculate best threshold on train set
    precisions, recalls, thresholds = precision_recall_curve(df.loc[train_mask, "Foehn"], prediction_probas[train_mask])
    best_threshold = thresholds[np.argmin(abs(precisions-recalls))-1]
    predictions = (prediction_probas > best_threshold).astype(int)
    
    # Plot precision and recall curves
    precisions, recalls, thresholds = precision_recall_curve(df.loc[test_mask, "Foehn"], prediction_probas[test_mask])
    f = plt.figure(figsize=(12,5))
    f.add_subplot(121)
    sns.lineplot(precisions, recalls)
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    f.add_subplot(122)
    sns.lineplot(np.append(thresholds, 1.0), precisions)
    sns.lineplot(np.append(thresholds, 1.0), recalls)
    plt.xlabel("Threshold")
    plt.legend(["Precision", "Recall"])
    
    # Print best threshold and performance on the test set
    print("--- Test set performance ---")
    print(f"Best threshold: {best_threshold}")
    print(confusion_matrix(df.loc[test_mask, "Foehn"], predictions[test_mask]))
    print(f'Accuracy: {accuracy_score(df.loc[test_mask, "Foehn"], predictions[test_mask])}')
    print(f'Precision: {precision_score(df.loc[test_mask, "Foehn"], predictions[test_mask])}')
    print(f'Recall: {recall_score(df.loc[test_mask, "Foehn"], predictions[test_mask])}')
    print(f'ROC-AUC: {roc_auc_score(df.loc[test_mask, "Foehn"], prediction_probas[test_mask])}')
    print(f'Log-Loss: {log_loss(df.loc[test_mask, "Foehn"], prediction_probas[test_mask])}')

    # Show 20 most important features
    df_ERA_feature_importances = pd.DataFrame({"feature_name": feature_names, "importance": best_model.feature_importances_}).sort_values(by="importance", ascending=False).reset_index(drop=True)
    display(df_ERA_feature_importances.head(20))
    
    return df_ERA_feature_importances

