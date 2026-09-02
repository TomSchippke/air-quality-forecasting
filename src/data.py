'''
This module is used for loading and preprocessing the data.
'''

import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
import torch 

def load_and_preprocess(location: str = "Aotizhongxin") -> pd.DataFrame:
    """
    Load and preprocess the data for a specific location.

    Args:
        location (str): The location to load the data for.

    Returns:
        pd.DataFrame: The preprocessed data.
    """

    df = pd.read_csv(f"../data/raw/PRSA_Data_20130301-20170228/PRSA_Data_{location}_20130301-20170228.csv")

    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Cyclical encoding so the model understands hour 23 wraps back to hour 0,
    # and month 12 wraps back to month 1.
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

    new_order = ['No', 'datetime', 'year', 'month_sin', 'month_cos', 'day', 'hour_sin', 'hour_cos', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM', 'station', 'PM2.5']
    df = df[new_order]

    # One-hot encode wind direction.
    df = pd.get_dummies(df, columns =['wd'], dtype=int, drop_first=True)

    df = df.ffill()

    return df


def add_lag_features(df: pd.DataFrame, target: str = "PM2.5", lags: list[int] = [1, 2, 3, 6, 12, 24]) -> pd.DataFrame:
    """
    Add lag features for the target variable.

    Args:
        df (pd.DataFrame): The data to add lag features to.
        target (str): The target variable.

    Returns:
        pd.DataFrame: The data with lag features.
    """

    df = df.copy()
    for l in lags: 
        df[f"{target}_lag_{l}"] = df[target].shift(l)
    
    return df

def temporal_split(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training, validation, and test sets based on temporal order.

    Args:
        df (pd.DataFrame): The data to split.
        train_ratio (float): The proportion of the data to use for training.
        val_ratio (float): The proportion of the data to use for validation.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The training, validation, and test sets.
    """
    
    split_index_train = int(len(df) * train_ratio)
    split_index_val = split_index_train + int(len(df) * val_ratio)

    train_idx = list(range(split_index_train))
    val_idx = list(range(split_index_train, split_index_val))
    test_idx = list(range(split_index_val, len(df)))

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    test_df = df.iloc[test_idx]
    
    return train_df, val_df, test_df

def split_X_y(df: pd.DataFrame, target: str = "PM2.5", to_drop: list[str] = ["datetime", "station", "No", "day", "year"], include_PM10: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into features (X) and target (y).

    Args:
        df (pd.DataFrame): The data to split.
        target (str): The target variable.
        to_drop (list[str]): The columns to drop.
        include_PM10 (bool): Whether to include PM10 as a feature.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The features and target.
    """

    y = df[target]

    if not include_PM10:
        X = df.drop(columns=to_drop+[target]+["PM10"])
    else: 
        X = df.drop(columns=to_drop+[target])

    return X, y

def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:    
    """
    Scale the features using StandardScaler.

    Args:
        X_train (pd.DataFrame): The training features.
        X_val (pd.DataFrame): The validation features.
        X_test (pd.DataFrame): The test features.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]: The scaled features.
    """
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, scaler


class AirQualityTorchDataset(torch.utils.data.Dataset):

    def __init__(self, 
        X: pd.DataFrame,
        y,
        input_window: int = 24,
        horizon: int = 12,
        
    ):
        self.X = torch.tensor(X.values, dtype = torch.float36)
        self.y = torch.tensor(y.values, dtype = torch.float36)
        self.input_window = input_window # size of the sequence used to predict the next value
        self.horizon = horizon # number of target
        # each time we use input_window timesteps to predict the next horizon timesteps

    def __len__(self) -> int:
        '''
        Return the total number of sliding windows we can fit during one epoch.

        Returns:
            int: The total number of sliding windows.
        '''
        return len(self.X) - self.input_window - self.horizon + 1
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Return the input window and the target window.

        Args:
            idx (int): The index of the input window.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The input window and the target window.
        '''
        x_seq = self.X[idx: idx + self.input_window]
        y_seq = self.y[idx + self.input_window : idx + self.input_window + self.horizon]
        return x_seq, y_seq
        
        



if __name__ == "__main__": 

    df = load_and_preprocess(
        location="Aotizhongxin"
    )
    df = add_lag_features(
        df=df,
        target="PM2.5",
        lags=[1,2,3,6,12,24]
    )

    train_df, val_df, test_df = temporal_split(df)
    X_train, y_train = split_X_y(train_df)
    X_val, y_val = split_X_y(val_df)
    X_test, y_test = split_X_y(test_df)

    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)


    train_torch_dataset = AirQualityTorchDataset(
        X_train, 
        y_train,
        input_window=48, # = 2 days of features
        horizon=6 # to predict the next 6 hours of PM2.5 
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_torch_dataset, 
        batch_size=32, 
        shuffle=True # shuffle the sliding windows for each epoch 
    )
 
    
    
