import pandas as pd
from data import (
    load_and_preprocess, 
    add_lag_features, 
    temporal_split, 
    split_X_y, 
    scale_features,
    AirQualityTorchDataset,
    AirQualityDataLoader
)
from models.lstm import LSTMForecaster, train_lstm
from models.baselines import train_random_forest


############################### LOAD AND PROCESS THE DATA ###############################

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

training_data_toch = AirQualityTorchDataset(
    X=X_train,
    y=y_train,
    input_window=48,
    horizon=6
)

val_data_toch = AirQualityTorchDataset(
    X=X_val,
    y=y_val,
    input_window=48,
    horizon=6
)

training_loader = AirQualityDataLoader(
    dataset=training_data_toch,
    batch_size=32,
    shuffle=True
)

val_loader = AirQualityDataLoader(
    dataset=val_data_toch,
    batch_size=32,
    shuffle=False
)

############################### MODEL TRAINING ###############################

## random forest
rf_model = train_random_forest(
    X_test_train=pd.concat([X_train, X_val], axis=0),
    y_test_train=pd.concat([y_train, y_val], axis=0),
    recall=False # turn True to avoid retraining if model already exist 
)


## LSTM model
lstm_model = LSTMForecaster(
    n_features=X_train.shape[1],
    hidden_size=64,
    num_layers=2,
    horizon=6,
    dropout=0.2,
)

history = train_lstm(
    model=lstm_model,
    train_loader=training_loader,
    val_loader=val_loader,
    n_epochs=50,
    lr=1e-3,
    patience=5,
    recall=False # turn True to avoid retraining if model already exist 
)