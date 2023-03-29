import argparse
import logging
import os

from ml_core import etl
from ml_core import settings as s

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


def generate() -> None:
    """
    Load the threadmill dataset, applies the dataframe preprocessing pipelane and stores result .csv in data directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="df_poc.csv",
        help="raw dataset to perform anomaly detection",
    )
    args = parser.parse_args()

    input_location = os.path.join(s.DATA_RAW, args.dataset)
    output_location = os.path.join(s.DATA_TRANSFORMED, args.dataset)
    etl.generate(input_location, output_location)


if __name__ == "__main__":
    generate()
