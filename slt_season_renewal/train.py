from sklearn.ensemble import HistGradientBoostingClassifier
import patsy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fan360_ds_utils.pandas_helpers import cast_to_numpy
from fan360_ds_utils import ModelPlotter
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    recall_score,
    precision_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)
from logzero import logger


def calc_feature_importances(estimator, x, y, num_features=10):
    """
    Function to print feature importances in descending order

    Parameters
    ----------
    estimator: sklearn.BaseEstimator
        Estimator to obtain feature importances from
    x: pd.DataFrame
        Features to compute importance of which must contain the column names
    y: pd.DataFrame
        Target dataframe
    num_features: int (default: 10)
        Number of features to print

    """
    logger.info("Calculating permutation importance")
    # this takes hours for large training datasets --> parameter reduces time
    if len(x) > 100000:
        parameter = 0.6
    else:
        parameter = len(x)
    importances = permutation_importance(
        estimator,
        x,
        y,
        n_repeats=5,
        random_state=0,
        n_jobs=-1,
        scoring="precision",
        max_samples=parameter,
    )
    features = pd.DataFrame(
        data={"feature": x.columns, "importance": np.abs(importances.importances_mean)}
    )
    features = features[features["importance"] > 0].sort_values(
        "importance", ascending=False
    )

    fig, ax = plt.subplots()
    sorted_idx = np.abs(importances.importances_mean).argsort()[-num_features:]
    ax.boxplot(
        importances.importances[sorted_idx].T,
        vert=False,
        labels=x.columns[sorted_idx],
    )
    ax.set_title("Permutation Importance of each feature")
    ax.set_ylabel("Features")
    fig.tight_layout()
    return fig, features

def scores_by_prob_level(threshold, probabilities, y_actual):
    """
    Function to print distribution of tags by changing probability score

    Parameters
    ----------
    threshold: list
        List of floats
    probabilities: list
        List containing probability of dropping for each observation
    y_actual: list
        List of observed results

    Returns
    -------
    None

    """
    test_predictions = [0] * len(probabilities)

    for probability in range(len(probabilities)):
        if probabilities[probability] >= threshold:
            test_predictions[probability] = 0
        else:
            test_predictions[probability] = 1

    logger.info("Scores if prob == {0:.2f}".format(threshold))
    logger.info(f"Test Precision score: {precision_score(y_actual, test_predictions)}")
    logger.info(f"Test Accuracy score: {accuracy_score(y_actual, test_predictions)}")
    logger.info(
        f"Test Confusion Matrix: \n {confusion_matrix(y_actual, test_predictions, labels=[True, False])}"
    )

def train_models(df, n_trials, training_timeout, model_logger):
    """
    Function to perform cross-validation grid-search, select best model, log to mlflow

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe containing all data necessary to train the models
    n_trials:
        max number of trials(experiments) trained in the optuna hyperparameter search
    training_timeout:
        max number of seconds for the optuna hyperparameter search
    model_logger: ModelLogger
        ModelLogger used to track model

    Returns
    -------
    None

    """
    logger.info(f"Begin train_model")
    df['target'] = df['bought_next_season']
    formula = """target ~  
            price +
            tickets_bought +
            purchase_days +
            rolling_seasons_bought +
            rolling_price_sum +
            rolling_price_mean +
            rolling_price_variance +
            last_years_price +
            price_yearly_delta +
            tickets_bought_last_year +
            tickets_bought_yearly_delta +
            last_years_purchase_days +
            purchase_days_yearly_delta +
            rolling_purchase_days_mean +
            rolling_purchase_days_variance +
            average_ticket_price +
            C(gender) +
            has_valid_first_name +
            has_valid_last_name +
            has_valid_phone +
            has_valid_postal +
            has_valid_email +
            distance_to_venue_miles +
            C(marital_status) +
            is_parent +
            household_size +
            C(education_level) +
            has_dob +
            C(income_level) +
            payment_amount +
            pledge_amount +
            count_of_mobile_comms +
            count_of_mobile_comms_viewed +
            count_of_mobile_comms_engaged +
            count_of_scans +
            num_touchpoints +
            num_connected_touchpoints +
            num_non_connected_touchpoints +
            cumulative_payment_amount +
            cumulative_season_payment_amount +
            cumulative_pledge_amount +
            cumulative_season_pledge_amount +
            cumulative_count_of_mobile_comms_viewed +
            cumulative_season_count_of_mobile_comms_viewed +
            cumulative_count_of_mobile_comms_engaged +
            cumulative_season_count_of_mobile_comms_engaged +
            cumulative_count_of_mobile_comms +
            cumulative_season_count_of_mobile_comms +
            cumulative_count_of_scans +
            cumulative_season_count_of_scans +
            cumulative_num_touchpoints +
            cumulative_season_num_touchpoints +
            cumulative_num_connected_touchpoints +
            cumulative_season_num_connected_touchpoints +
            cumulative_num_non_connected_touchpoints +
            cumulative_season_num_non_connected_touchpoints +
            rolling_mean_freq_of_purchase +
            freq_distance_from_average_purchase_freq 
        """


    y,X = patsy.dmatrices(formula, df, return_type = 'dataframe',NA_action=patsy.NAAction(NA_types=[]))
    # X = X.fillna(0) Don't need this since running a tree model that handles nulls
    y_ravel = np.ravel(y)

    model = HistGradientBoostingClassifier(class_weight='balanced',early_stopping = True, n_iter_no_change=15)

    logger.info(f"Model Evaluation Model Fit")
    results = model.fit(X,y_ravel)
    df['model_eval_prediction'] = results.predict(X)
    df['model_eval_probability'] = results.predict_proba(X)[:,1]



    # split dataset to training/testing sets
    test_date = df['season'].max()-2
    train_indices = df.loc[~(df['season'].isin(range(test_date,df['season'].max()+1)))].index
    test_indices = df.loc[(df['season'].isin([test_date]))].index
    
    X_train = X.loc[X.index.isin(train_indices)]
    X_test = X.loc[X.index.isin(test_indices)]
    y_train = y.loc[y.index.isin(train_indices)]
    y_test = y.loc[y.index.isin(test_indices)]

    logger.info(f"Model Test Fit")
    y_train_ravel = np.ravel(y_train)
    train_results = model.fit(X_train,y_train_ravel)
    test_predictions = train_results.predict(X_test)
    test_probability = train_results.predict_proba(X_test)

    test_df = df.loc[df.index.isin(X_test.index)]
    test_df['probability'] = test_probability[:,1]
    test_df['prediction'] = test_predictions

    #TODO: Add more robust testing validation (this is somewhat biased and we probably need a better way to test since we are always based off of last year's predicitons.)
    #Unfortunately, we don't really have any better way to do that right now, especially since there is such limited data.

    logger.info(f"Calculate scores for Model Eval")
    # calculate model evaluation metrics
    model_eval_recall = recall_score(df['target'], df['model_eval_prediction'])
    model_eval_precision = precision_score(df['target'], df['model_eval_prediction'])
    model_eval_accuracy = accuracy_score(df['target'], df['model_eval_prediction'])
    model_eval_roc_auc = roc_auc_score(df['target'], df['model_eval_prediction'])

    logger.info(f"Calculate scores for Model Test")
    # calculate test evaluation metrics
    test_eval_recall = recall_score(y_test, test_df['prediction'])
    test_eval_precision = precision_score(y_test, test_df['prediction'])
    test_eval_accuracy = accuracy_score(y_test, test_df['prediction'])
    test_eval_roc_auc = roc_auc_score(y_test, test_df['prediction'])

    # log metrics
    logger.info(f"Log score metrics")
    model_logger.log_metric(key="model_eval_recall", value=model_eval_recall)
    model_logger.log_metric(key="model_eval_precision", value=model_eval_precision)
    model_logger.log_metric(key="model_eval_accuracy", value=model_eval_accuracy)
    model_logger.log_metric(key="model_eval_auc", value=model_eval_roc_auc)
    model_logger.log_metric(key="test_eval_recall", value=test_eval_recall)
    model_logger.log_metric(key="test_eval_precision", value=test_eval_precision)
    model_logger.log_metric(key="test_eval_accuracy", value=test_eval_accuracy)
    model_logger.log_metric(key="test_eval_auc", value=test_eval_roc_auc)

    logger.info(f"Start Model Logger")
    # log visualizations
    plotter = ModelPlotter(
        estimator=model,
        x_train=X_train,
        y_train=y_train_ravel,
        x_test=X_test,
        y_test=y_test,
        scoring_method="precision",
        labels=[True, False],
        fillna=0.65,
        dimensionality_reduction="pca",
        model_logger=model_logger,
    )
    plotter.generate_plots()

    logger.info(f"Calculate Feature Importance")
    fig, importances = calc_feature_importances(
        model,
        x=X_test,
        y=y_test,
    )
    model_logger.log_run_artifact(
        artifact_name="feature_importances.png",
        artifact=fig,
    )

    model_logger.end_run()

    # log artifacts/model
    fig, importances = calc_feature_importances(
        model,
        x=X,
        y=y,
    )
    model_logger.log_run_artifact(artifact_name="feature_importances.png", artifact=fig)
    importances["org_mnemonic"] = model_logger.current_experiment.name.split("-")[1]
    model_logger.s3_manager.write_df(
        df=importances,
        s3_path=f"{model_logger.artifact_uri_base}/{model_logger.current_experiment.name}/artifacts/permutation_importances.parquet",
        partition_cols=["org_mnemonic"],
    )

    # save model
    logger.info(f"Saving Model")
    model_logger.log_model(model, register=True, train=X, target=df['target'])
    model_logger.end_run()

def train(model_logger, n_trials, training_timeout):
    """
    Function to manage model training

    Parameters
    ----------
    model_logger: ModelLogger
        ModelLogger used to track model
    n_trials: int
        max number of trials used in optuna hyperparameter searching
    training_timeout: int
        max number of seconds used in optuna hyperparameter searching
    """
    train_df = model_logger.load_experiment_artifact("train_df.parquet")

    logger.info(f"Training models for {model_logger.current_experiment.name}")

    train_models(train_df, n_trials, training_timeout, model_logger)

    logger.info("Training complete")