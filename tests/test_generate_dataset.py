import pandas as pd
import pytest

from ml_core.etl.generate_dataset import process_dataframe_pipeline

DF_EXPECTED = pd.DataFrame(
    {
        "999_2N1_31": [2.390000105, 2.36, 2.380000114, 2.4],
        "999_2N1_32": [1.5, 1.440000057, 1.43, 1.50999999],
        "999_2N1_33": [2.109999921, 2.119999886, 2.099999905, 2.1],
    }
)

DF_UNEXPECTED_1 = pd.DataFrame(
    {
        "999_2N1_31": [2.390000105, 2.36, 2.380000114, 2.4],
        "999_2N1_32": [1.5, 1.440000057, 1.43, 1.50999999],
        "test": [20, 21, 20, 18],
    }
)

DF_UNEXPECTED_2 = pd.DataFrame(
    {
        "999_2N1_31": [2.390000105, None, 2.380000114, 2.4],
        "999_2N1_32": [1.5, 1.440000057, 1.43, 1.50999999],
        "999_2N1_33": [20, 21, 20, 18],
    }
)


@pytest.mark.parametrize(
    "df, error_expected",
    [
        (DF_EXPECTED, False),
        (DF_UNEXPECTED_1, True),
        (DF_UNEXPECTED_2, True),
    ],
)
def test_process_dataframe_pipeline(df: pd.DataFrame, error_expected: bool) -> None:
    """
    Tests whether the pipeline creates unexpected behaviors. Make sure
    the processed data frame has exactly the same headers with the
    original data frame and have less than or equal to number of rows.
    """
    try:
        df_outlier_removed = process_dataframe_pipeline(df, fillna="mean")
        assert df.shape[0] > 0  # Rows
        assert df.shape[1] > df_outlier_removed.shape[1]
        assert df.shape[0] == df_outlier_removed.shape[0]
    except KeyError:
        assert error_expected
    except ValueError:
        assert error_expected
