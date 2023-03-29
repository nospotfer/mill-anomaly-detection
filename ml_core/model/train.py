"""
Anomaly Detection training procedure on the Vertical Mill dataset provided.

The dataset is provided by insus.ch
"""
import logging
import os

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from ml_core import settings as s
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


def train(
    dataset_loc: str,
    model_dir: str,
    model_name: str = "if",
    original_dataset_loc: str = "",
    evaluate: bool = True,
) -> None:
    """
    Trains the unsupervised anomaly detection model.
    The training results with the accompanying models is
    saved in ./models/

    Parameters:
        dataset_loc (str): the dataset path on which we want to train

        model_dir (str): directory of the serialized ml models

        model_name (str): the model_name that we want to use as a save
             default:
                "if": isolation forest

        evaluate (bool): whether to evaluate the model performance on the training set
    Returns:
        None

    """
    # loading data
    df = pd.read_csv(dataset_loc)

    # Set training data
    X_train = df

    # pre-processing
    # RobustScaler() is a method for scaling numerical features in a way that is robust to outliers.
    # It can help ensure that our Isolation Forest is robust to outliers and able to accurately detect anomalies in the data.
    scaler = RobustScaler()

    # In this specific example we will use an off-the-shelf Isolation Forest algorithm
    # No information about columns is provided on this take-home test, nor true labels for data.
    # Therefore, we will use an unsupervised approach to detect anomalies
    model = IsolationForest(random_state=s.RANDOM_STATE)

    # create pipeline
    pipeline = make_pipeline(scaler, model)

    # training
    # model.fit(X_train)
    pipeline.fit(X_train)

    if evaluate:
        predict(pipeline, X_train, original_dataset_loc)
    logger.info(f"Model: {pipeline.__class__.__name__}")

    # Evaluate model performance should go here if we had labels.
    # score = evaluate_model(pipeline, y_train)
    score = 100.0

    # Serialize and dump trained pipeline to disk
    pred_result = {
        "model_name": model_name,
        "score": score,
        "deserialized_model": pipeline,
    }

    model_location = os.path.join(model_dir, model_name) + ".joblib"
    with open(model_location, "wb") as f:
        # Serialize pipeline and compress it with the max factor 9
        joblib.dump(pred_result, f, compress=9)


def predict(pipeline, X_train, original_dataset_loc):
    anomalies = pipeline.predict(X_train)
    anomaly_scores = pipeline.decision_function(X_train)
    score_samples = pipeline.score_samples(X_train)
    df_raw = pd.read_csv(original_dataset_loc)
    df_raw["anomaly"] = anomalies
    df_raw["anomaly_scores"] = anomaly_scores
    df_raw["anomaly_scores_moving_avg_6"] = (
        df_raw["anomaly_scores"].rolling(window=5).mean()
    )
    df_raw["anomaly_scores_moving_avg_12"] = (
        df_raw["anomaly_scores"].rolling(window=12).mean()
    )
    df_raw["score_samples"] = score_samples
    df_raw = df_raw.rename(columns={"Unnamed: 0": "datetime_str"})
    df_raw["datetime"] = df_raw["datetime_str"].apply(
        lambda x: pd.to_datetime(x, format="%Y-%m-%d %H:%M")
    )
    df_raw.to_csv(
        os.path.join(s.DATA_PREDICTIONS, "df_poc_predictions.csv"), index=False
    )

    ## PLOT
    df_raw = df_raw.set_index("datetime")
    fig, ax = plt.subplots(figsize=(18, 6))  # set the figure size

    df_raw[["anomaly_scores_moving_avg_6", "anomaly_scores_moving_avg_12"]].plot(ax=ax)
    ax.set_xlabel("Datetime")  # set the x-axis label
    ax.tick_params(
        axis="x", which="major", labelsize=10
    )  # set the tick label font size
    plt.ylabel("Anomaly Scores with moving average 6 and 12")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(s.DATA_PREDICTIONS, "anomaly_scores_moving_avg_6_12.png"))

    fig, ax = plt.subplots(figsize=(18, 6))  # set the figure size
    df_raw[["anomaly_scores"]].plot(ax=ax)
    ax.set_xlabel("Datetime")  # set the x-axis label
    ax.tick_params(
        axis="x", which="major", labelsize=10
    )  # set the tick label font size
    plt.ylabel("Anomaly Scores")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(s.DATA_PREDICTIONS, "anomaly_scores.png"))


def evaluate_model(pipeline, y_train) -> float:
    """
    Evaluate the model performance on the training set
    If we had labels we could use the model to evaluate the performance.
    We should use the Area under the precision-recall curve (AUC-PR) instead of the AUC-ROC since it is better for
    imbalanced dataset with a small number of anomalous data points.
    """
    return 100.0
