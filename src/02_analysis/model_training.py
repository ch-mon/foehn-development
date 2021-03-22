from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb


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






class ConstrainedXGBoost:
    
    def __init__(self, df_ERA, df_CESMp, features, train_mask, test_mask, params, initial_boosting_rounds):
        print("Loading data ...")
        D_train = xgb.DMatrix(df_ERA.loc[train_mask, features],
                              label=df_ERA.loc[train_mask, "Foehn"],
                              weight=df_ERA.loc[train_mask, "Foehn"] * 20 + 1)
        print("Fitting global model ...")
        model = xgb.Booster(params, [D_train])
        model = xgb.train(params, dtrain=D_train, num_boost_round=initial_boosting_rounds, xgb_model=model)

        print("Storing global model ...")
        self.global_model = model.copy()
        self.features = features
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.df_ERA = df_ERA
        self.df_CESMp = df_CESMp
        self.local_models = dict()
        self.monthly_thresholds = dict()
        self.precision_scores = dict()
        self.recall_scores = dict()
        self.f1_scores = dict()
        self.total_confusion_matrix = np.zeros((2, 2))
        self.most_important_features = dict()

        print("Model ready to use ...")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def fit_for_month(self, month):
        """
        Continue fitting the global model on now monthly data
        @param month: The month of interest (1-12)
        @type month: int
        @return: Fitted model
        @rtype: xgb.Booster
        """

        def custom_loss(prediction: np.ndarray, dtrain: xgb.DMatrix, ERA_len, CESM_len, alpha=60000):
            """
            Specify a loss function which takes an constraint objective into consideration and returns gradient and
            hessian for the training. This objective (Squared Mean Error) is then optimized on the unlabeled CESM data.
            @param prediction: Prediction values for stacked ERAI and CESMp dataframes
            @type prediction: np.ndarray
            @param dtrain: Training set (features and labels)
            @type dtrain: xgb.DMatrix
            @param ERA_len: Number of ERAI samples in dtrain
            @type ERA_len: int
            @param CESM_len: umber of CESMp samples in dtrain (after ERAI samples)
            @type CESM_len: int
            @param alpha: Constraint parameter
            @type alpha: int
            @return: Gradient and hessian of prediction for ERAI and CESM samples
            @rtype: np.ndarray, np.ndarray
            """

            # Get ERAI labels
            y_ERA = dtrain.get_label()[:ERA_len]

            # Split predictions into ERAI and CESM
            prediction = self.sigmoid(prediction)
            pred_ERA = prediction[:ERA_len]
            pred_CESM = prediction[ERA_len:]

            # Calculate log loss gradient and hessian for ERAI samples
            grad_logloss = pred_ERA - y_ERA
            hess_logloss = pred_ERA * (1.0 - pred_ERA)

            # Calculate SME gradient and hessian for CESM samples
            grad_SME = 2 * alpha / CESM_len * pred_CESM * (1 - pred_CESM) * (np.mean(pred_CESM) - np.mean(y_ERA))
            hess_SME = 2 * alpha / CESM_len * pred_CESM * (1 - pred_CESM) * (
                        (1 - pred_CESM) * (np.mean(pred_CESM) - np.mean(y_ERA)) -
                        pred_CESM * (np.mean(pred_CESM) - np.mean(y_ERA)) +
                        pred_CESM * (1 - pred_CESM) / CESM_len
                        )

            # Build full gradient for all samples
            grad = np.zeros(len(prediction))
            grad[:ERA_len] = grad_logloss
            grad[ERA_len:] = grad_SME

            # Build full hessian for all samples
            hess = np.zeros(len(prediction))
            hess[:ERA_len] = hess_logloss
            hess[ERA_len:] = hess_SME

            return grad, hess

        # Define masks to address specific parts of data set
        train_mask_month = self.train_mask & (self.df_ERA["date"].dt.month == month)
        month_mask_CESMp = (self.df_CESMp["date"].dt.month == month)

        # Build training dataset from ERA and CESMp
        df_ERA_CESMp = pd.concat([self.df_ERA.loc[train_mask_month, self.features + ["Foehn"]], self.df_CESMp.loc[month_mask_CESMp, self.features]], axis=0, ignore_index=True)
        D_train = xgb.DMatrix(df_ERA_CESMp[self.features], label=df_ERA_CESMp["Foehn"], weight=(df_ERA_CESMp["Foehn"]*20+1))
        ERA_len = train_mask_month.sum()
        CESM_len = month_mask_CESMp.sum()

        # Train model
        model = self.global_model.copy()
        for _ in range(190):
            pred = model.predict(D_train)
            g, h = custom_loss(pred, D_train, ERA_len=ERA_len, CESM_len=CESM_len)
            model.boost(D_train, g, h)

        self.local_models[month] = model.copy()


    def evaluate_for_month(self, month):
        """
        Evaluate the performance of a trained classifier for a given month.
        @param month: Month of interest (1-12)
        @type month: int
        @return:
        @rtype:
        """

        # Define ERA masks
        test_mask_month = self.test_mask & (self.df_ERA["date"].dt.month == month)
        month_mask_ERA = (self.df_ERA["date"].dt.month == month)

        # Make prediction on whole ERAI data
        D_ERA = xgb.DMatrix(self.df_ERA.loc[month_mask_ERA, self.features])
        self.df_ERA.loc[month_mask_ERA, "prediction_proba"] = self.sigmoid(self.local_models[month].predict(D_ERA))

        # Calculate metrics on test set & identify best threshold
        precisions, recalls, thresholds = precision_recall_curve(self.df_ERA.loc[test_mask_month, "Foehn"], self.df_ERA.loc[test_mask_month, "prediction_proba"])
        self.monthly_thresholds[month] = thresholds[np.argmin(abs(precisions - recalls)) - 1]

        # Make final prediction for ERA
        self.df_ERA.loc[month_mask_ERA, "prediction"] = (self.df_ERA.loc[month_mask_ERA, "prediction_proba"] > self.monthly_thresholds[month]).astype(int)

        # Calculate evaluation scores
        self.precision_scores[month] = precision_score(self.df_ERA.loc[test_mask_month, "Foehn"], self.df_ERA.loc[test_mask_month, "prediction"])
        self.recall_scores[month] = recall_score(self.df_ERA.loc[test_mask_month, "Foehn"], self.df_ERA.loc[test_mask_month, "prediction"])
        self.f1_scores[month] = f1_score(self.df_ERA.loc[test_mask_month, "Foehn"], self.df_ERA.loc[test_mask_month, "prediction"])
        self.total_confusion_matrix += confusion_matrix(self.df_ERA.loc[test_mask_month, "Foehn"], self.df_ERA.loc[test_mask_month, "prediction"])

        # Print evaluation scores
        print(f"Best threshold: {self.monthly_thresholds[month]}")
        print(f'Precision: {self.precision_scores[month]}')
        print(f'Recall: {self.f1_scores[month]}')
        print(confusion_matrix(self.df_ERA.loc[test_mask_month, "Foehn"], self.df_ERA.loc[test_mask_month, "prediction"]))

        # Aggregate feature importances for all months
        importances_month = pd.DataFrame.from_dict(self.local_models[month].get_score(importance_type='gain'), orient="index", columns=["importance"]).sort_values(by="importance", ascending=False)
        for row in importances_month.head(30).iterrows():  # Only consider 30 most important features per month
            if row[0] in self.most_important_features.keys():  # Add feature importance if feature already exists
                self.most_important_features[row[0]] += row[1][0]
            else:  # Otherwise create entry for this feature
                self.most_important_features[row[0]] = row[1][0]

    def predict_proba(self):
        pass
    
    def predict_for_month(self, df, month):
        """
        Predict on ERAI, CESMp, or CESMf for a given month
        @param df: ERAI, CESMp, or CESMf dataframe with features used during training
        @type df: pd.DataFrame
        @param month: Month of interest
        @type month: int
        @return: Dataframe with original and prediction values
        @rtype: pd.DataFrame
        """

        # Define mask for given month
        month_mask = (df["date"].dt.month == month)

        # Create dataset and classify with learned parameters
        dataset = xgb.DMatrix(df.loc[month_mask, self.features])
        df.loc[month_mask, "prediction_proba"] = self.sigmoid(self.local_models[month].predict(dataset))
        df.loc[month_mask, "prediction"] = (df.loc[month_mask, "prediction_proba"] > self.monthly_thresholds[month]).astype(int)

        return df

    def predict_for_all_months(self, df):
        """
        Predict for all months in a dataset. Wrapper for self.predict_for_month.
        @param df: ERAI, CESMp, or CESMf dataframe with features used during training
        @type df: pd.DataFrame
        @return: Dataframe with original and prediction values
        @rtype: pd.DataFrame
        """

        # Loop over all months
        for month in range(1, 12):
            df = self.predict_for_month(df, month)
        return df


def create_stacked_dataframe(df_ERA, df_CESMp, df_CESMf):
    """
    Stack observed foehn data, predicted ERAI data, CESMp data, and CESMf data for easier processing
    @param df_ERA: ERAI data
    @type df_ERA: pd.DataFrame
    @param df_CESMp: CESMp data
    @type df_CESMp: pd.DataFrame
    @param df_CESMf: CESMf data
    @type df_CESMf: pd.DataFrame
    @return: Stacked dataframe
    @rtype: pd.DataFrame
    """
    # Add more more information to each dataframe before merge
    df_ERA["dataset"] = "ERA"
    df_ERA["ensemble"] = "ERA"
    df_CESMp["dataset"] = "CESMp"
    df_CESMf["dataset"] = "CESMf"

    # Create dataframe for actually observed foehn cases
    df_foehn = df_ERA.loc[:, ["date", "Foehn"]]
    df_foehn["dataset"] = "observed_foehn"
    df_foehn["ensemble"] = "observed_foehn"
    df_foehn.rename({"Foehn": "prediction"}, axis=1, inplace=True)

    # Concat all observed foehn, ERAI, CESMp, and CESMf prediction
    df_foehn_ERA_CESMp_CESMf = pd.concat([df_foehn,
                                          df_ERA[["date", "prediction", "dataset", "ensemble"]],
                                          df_CESMp[["date", "prediction", "dataset", "ensemble"]],
                                          df_CESMf[["date", "prediction", "dataset", "ensemble"]]],
                                         axis=0,
                                         ignore_index=True)

    # Create month and year column
    df_foehn_ERA_CESMp_CESMf["month"] = df_foehn_ERA_CESMp_CESMf["date"].dt.month
    df_foehn_ERA_CESMp_CESMf["year"] = df_foehn_ERA_CESMp_CESMf["date"].dt.year

    return df_foehn_ERA_CESMp_CESMf


