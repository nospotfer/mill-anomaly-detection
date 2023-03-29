import argparse
import os

from ml_core import model
from ml_core import settings as s


def train() -> None:
    """
    Load the threadmill dataset, applies the dataframe preprocessing pipelane and stores result .csv in data directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="df_poc.csv",
        help="raw dataset to perform anomaly detection",
    )
    parser.add_argument(
        "--model-name",
        default="if",
        help="the serialized model name. default if " "referring to Isolation Forest",
    )
    args = parser.parse_args()
    transformed_data_dir = os.path.join(s.DATA_TRANSFORMED, args.dataset)
    original_data_dir = os.path.join(s.DATA_RAW, args.dataset)
    model.train(
        transformed_data_dir,
        s.MODEL_DIR,
        args.model_name,
        original_data_dir,
        evaluate=True,
    )


if __name__ == "__main__":
    train()
