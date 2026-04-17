import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class TestDataLoader:
    """
    Offline evaluation data loader.
    """

    def __init__(self, file_path="./data/log.parquet"):
        """
        Initialize the data loader.
        Args:
            file_path (str): The path to the training data file (parquet format).

        """
        self.file_path = file_path
        self.raw_data = self._get_raw_data()
        self.keys, self.test_dict = self._get_test_data_dict()

    def _get_raw_data(self):
        """
        Read raw data from a parquet file.

        Returns:
            pd.DataFrame: The raw data as a DataFrame.
        """
        return pd.read_parquet(self.file_path)

    def _get_test_data_dict(self):
        """
        Group and sort the raw data by deliveryPeriodIndex and advertiserNumber.

        Returns:
            list: A list of group keys.
            dict: A dictionary with grouped data.

        """
        grouped_data = self.raw_data.sort_values('timeStepIndex').groupby(['deliveryPeriodIndex', 'advertiserNumber'])
        data_dict = {key: group for key, group in grouped_data}
        return list(data_dict.keys()), data_dict

    def mock_data(self, key):
        """
        Get training data based on deliveryPeriodIndex and advertiserNumber, and construct the test data.
        """
        data = self.test_dict[key]
        pValues = data.groupby('timeStepIndex')['pValue'].apply(list).apply(np.array).tolist()
        pValueSigmas = data.groupby('timeStepIndex')['pValueSigma'].apply(list).apply(np.array).tolist()
        leastWinningCosts = data.groupby('timeStepIndex')['leastWinningCost'].apply(list).apply(np.array).tolist()
        num_timeStepIndex = len(pValues)
        return num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts


if __name__ == '__main__':
    pass
