from scipy.optimize._trustregion_constr import equality_constrained_sqp
import torch
import json
import numpy as np
import sys
from src.data import (
    load_and_preprocess, 
    add_lag_features, 
    temporal_split, 
    split_X_y_seq, 
    split_X_y_rf,
    scale_features,
    scale_target,
    AirQualityLSTMDataset
)
from torch.utils.data import DataLoader
from src.models.lstm import LSTMForecaster, train_lstm, tune_lstm
from src.models.transformer import TransformerForecaster, train_transformer, tune_transformer
from src.models.baselines import train_random_forest
from sklearn.metrics import mean_squared_error, mean_absolute_error

############################### GLOBAL PARAMETERS ###############################

HORIZON = 6
INCLUDE_PM10 = False
TUNE_LSTM = True
TUNE_TRANSFORMER = True
FEATURE_TO_DROP = ["datetime", "station", "No", "day", "year"]


############################### LOAD AND PROCESS THE DATA ###############################

df = load_and_preprocess(
    location="Aotizhongxin"
)

df = add_lag_features(
    df=df,
    target="PM2.5",
    lags=[1,2,3,6,12,24],
    horizon=HORIZON
)

train_df, val_df, test_df = temporal_split(df)

train_df = train_df.dropna().reset_index(drop=True)
val_df = val_df.dropna().reset_index(drop=True)
test_df = test_df.dropna().reset_index(drop=True)


### seq data
lag_columns = [f"PM2.5_lag_{l}" for l in [1, 2, 3, 6, 12, 24]]
horizon_columns = [f"PM2.5_t+{l}" for l in range(1, HORIZON + 1)]
X_train_seq, y_train_seq = split_X_y_seq(train_df, to_drop=FEATURE_TO_DROP + lag_columns + horizon_columns, include_PM10=INCLUDE_PM10)
X_val_seq, y_val_seq = split_X_y_seq(val_df, to_drop=FEATURE_TO_DROP + lag_columns + horizon_columns, include_PM10=INCLUDE_PM10)
X_test_seq, y_test_seq = split_X_y_seq(test_df, to_drop=FEATURE_TO_DROP + lag_columns + horizon_columns, include_PM10=INCLUDE_PM10)

### rf data
X_train_rf, y_train_rf = split_X_y_rf(train_df, horizon_columns=horizon_columns, include_PM10=INCLUDE_PM10)
X_val_rf, y_val_rf = split_X_y_rf(val_df, horizon_columns=horizon_columns, include_PM10=INCLUDE_PM10)
X_test_rf, y_test_rf = split_X_y_rf(test_df, horizon_columns=horizon_columns, include_PM10=INCLUDE_PM10)


X_train_rf, X_val_rf, X_test_rf, scaler_rf = scale_features(X_train_rf, X_val_rf, X_test_rf)
X_train_seq, X_val_seq, X_test_seq, scaler_seq = scale_features(X_train_seq, X_val_seq, X_test_seq)

y_train_rf, y_val_rf, y_test_rf, target_scaler_rf = scale_target(y_train_rf, y_val_rf, y_test_rf)
y_train_seq, y_val_seq, y_test_seq, target_scaler_seq = scale_target(y_train_seq, y_val_seq, y_test_seq)


training_data_toch = AirQualityLSTMDataset(
    X=X_train_seq,
    y=y_train_seq,
    input_window=48,
    horizon=HORIZON
)

val_data_toch = AirQualityLSTMDataset(
    X=X_val_seq,
    y=y_val_seq,
    input_window=48,
    horizon=HORIZON
)

test_data_toch = AirQualityLSTMDataset(
    X=X_test_seq,
    y=y_test_seq,
    input_window=48,
    horizon=HORIZON
)

training_loader = DataLoader(
    dataset=training_data_toch,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    dataset=val_data_toch,
    batch_size=16,
    shuffle=False
)

test_loader = DataLoader(
    dataset=test_data_toch,
    batch_size=16,
    shuffle=False
)

print(f"=========================================")
print(f"Random Forest Features are ({len(scaler_rf.feature_names_in_)}) : {list(scaler_rf.feature_names_in_)}")
print(f"LSTM / Transformers Features are ({len(scaler_seq.feature_names_in_)}) : {list(scaler_seq.feature_names_in_)}")
print(f"=========================================")
print(f"End of data preprocessing")
print(f"=========================================")


############################### MODEL TRAINING ###############################

## Random forest model
rf_model = train_random_forest(
    X_test_train=np.concatenate([X_train_rf, X_val_rf], axis=0),
    y_test_train=np.concatenate([y_train_rf, y_val_rf], axis=0),
    recall=True # Force retraining since we changed the target to 6 horizons
)

## LSTM model
if TUNE_LSTM:
    best_lstm_params = tune_lstm(
        train_loader=training_loader,
        val_loader=val_loader,
        n_features=X_train_seq.shape[1],
        horizon=HORIZON,
        n_trials=30
    )
    lstm_model = LSTMForecaster(
        n_features=X_train_seq.shape[1],
        hidden_size=best_lstm_params["hidden_size"],
        num_layers=best_lstm_params["num_layers"],
        horizon=HORIZON,
        dropout=best_lstm_params["dropout"],
    )
    history = train_lstm(
        model=lstm_model,
        train_loader=training_loader,
        val_loader=val_loader,
        n_epochs=100,
        lr=best_lstm_params["lr"],
        patience=10,
        recall=False
    )
else:
    lstm_model = LSTMForecaster(
        n_features=X_train_seq.shape[1],
        hidden_size=128,
        num_layers=2,
        horizon=HORIZON,
        dropout=0.4,
    )
    
    history = train_lstm(
        model=lstm_model,
        train_loader=training_loader,
        val_loader=val_loader,
        n_epochs=100,
        lr=1e-4,
        patience=10,
        recall=True # turn True to avoid retraining if model already exist 
    )


## Transformer model
if TUNE_TRANSFORMER:
    best_trans_params = tune_transformer(
        train_loader=training_loader,
        val_loader=val_loader,
        n_features=X_train_seq.shape[1],
        horizon=HORIZON,
        n_trials=30
    )
    transformer_model = TransformerForecaster(
        n_features=X_train_seq.shape[1],
        d_model=best_trans_params["d_model"],
        nhead=best_trans_params["nhead"],
        num_layers=best_trans_params["num_layers"],
        dropout=best_trans_params["dropout"],
        horizon=HORIZON
    )
    history = train_transformer(
        model=transformer_model,
        train_loader=training_loader,
        val_loader=val_loader,
        n_epochs=100,
        lr=best_trans_params["lr"],
        patience=10,
        recall=False
    )
else:
    transformer_model = TransformerForecaster(
        n_features=X_train_seq.shape[1],
        d_model=128,
        nhead=8,        
        num_layers=2,        
        dropout=0.3,
        horizon=HORIZON
    )
    
    history = train_transformer(
        model=transformer_model,
        train_loader=training_loader,
        val_loader=val_loader,
        n_epochs=100,
        lr=1e-5,
        patience=10,
        recall=True 
    )


############################### QUICK MODEL EVALUATION ###############################

print("=========================================")
print("Starting model evaluation...")
print("=========================================")


metrics = {}

def compute_and_save(y_true, y_pred, model_name: str, scaler=None) -> dict:
    '''
    Compute RMSE and MAE and save the predictions and true values.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
        model_name (str): Name of the model.
        scaler: The scaler used to unscale the data.

    Returns:
        dict: Dictionary containing RMSE and MAE.
    '''
    import os
    os.makedirs("results/predictions", exist_ok=True)
    
    if scaler is not None:
        if scaler.n_features_in_ == 1 and len(y_true.shape) == 2 and y_true.shape[1] > 1:
            orig_shape_true = y_true.shape
            orig_shape_pred = y_pred.shape
            y_true = scaler.inverse_transform(y_true.flatten().reshape(-1, 1)).reshape(orig_shape_true)
            y_pred = scaler.inverse_transform(y_pred.flatten().reshape(-1, 1)).reshape(orig_shape_pred)
        else:
            y_true = scaler.inverse_transform(y_true)
            y_pred = scaler.inverse_transform(y_pred)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    np.save(f"results/predictions/{model_name}_y_true.npy", y_true)
    np.save(f"results/predictions/{model_name}_y_pred.npy", y_pred)
    return {"RMSE": rmse, "MAE": mae}


### Random Forest evaluation
y_pred_rf = rf_model.predict(X_test_rf)
metrics["random_forest"] = compute_and_save(y_test_rf, y_pred_rf, "rf", target_scaler_rf)


### Naive Baseline evaluation
y_true_naive = test_df[horizon_columns].values
y_pred_naive = np.tile(test_df[["PM2.5"]].values, (1, HORIZON))
metrics["naive"] = compute_and_save(y_true_naive, y_pred_naive, "naive", scaler=None)


### LSTM evaluation
device = "cuda" if torch.cuda.is_available() else "cpu"
lstm_model.eval()
preds, trues = [], []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        preds.append(lstm_model(x_batch).numpy())
        trues.append(y_batch.numpy())

y_pred_lstm = np.concatenate(preds)
y_true_lstm = np.concatenate(trues)
metrics["lstm"] = compute_and_save(y_true_lstm, y_pred_lstm, "lstm", target_scaler_seq)
    

### Transformer evaluation
device = "cuda" if torch.cuda.is_available() else "cpu"
transformer_model.eval()
preds, trues = [], []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        preds.append(transformer_model(x_batch).cpu().numpy())
        trues.append(y_batch.numpy())

y_pred_transformer = np.concatenate(preds)
y_true_transformer = np.concatenate(trues)
metrics["transformer"] = compute_and_save(y_true_transformer, y_pred_transformer, "transformer", target_scaler_seq)
    

### Save 
with open("results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("=========================================")
print("Metrics : ")
print("=========================================")

for model, metric in metrics.items():
    print(f"{model}: RMSE={metric['RMSE']}, MAE={metric['MAE']}")

print("=========================================")
print("End of model evaluation")
print("=========================================")
