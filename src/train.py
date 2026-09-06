from scipy.optimize._trustregion_constr import equality_constrained_sqp
import torch
import json
import numpy as np
import sys
import os
import joblib
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
TUNE_LSTM = False
TUNE_TRANSFORMER = False
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

# Apply log1p transformation for sequence models
y_train_seq = np.log1p(y_train_seq)
y_val_seq = np.log1p(y_val_seq)
y_test_seq = np.log1p(y_test_seq)
y_train_seq, y_val_seq, y_test_seq, target_scaler_seq = scale_target(y_train_seq, y_val_seq, y_test_seq)

os.makedirs("results/scalers", exist_ok=True)
joblib.dump(scaler_rf, "results/scalers/scaler_rf.joblib")
joblib.dump(scaler_seq, "results/scalers/scaler_seq.joblib")
joblib.dump(target_scaler_rf, "results/scalers/target_scaler_rf.joblib")
joblib.dump(target_scaler_seq, "results/scalers/target_scaler_seq.joblib")

# Datasets and Loaders are now built dynamically based on the tuned window_size

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
        X_train=X_train_seq,
        y_train=y_train_seq,
        X_val=X_val_seq,
        y_val=y_val_seq,
        n_features=X_train_seq.shape[1],
        horizon=HORIZON,
        n_trials=30
    )
    with open("results/lstm_best_params.json", "w") as f:
        json.dump(best_lstm_params, f, indent=4)
    lstm_model = LSTMForecaster(
        n_features=X_train_seq.shape[1],
        hidden_size=best_lstm_params["hidden_size"],
        num_layers=best_lstm_params["num_layers"],
        horizon=HORIZON,
        dropout=best_lstm_params["dropout"],
    )
    lstm_window_size = best_lstm_params.get("window_size", 48)
    training_data_lstm = AirQualityLSTMDataset(X=X_train_seq, y=y_train_seq, input_window=lstm_window_size, horizon=HORIZON)
    val_data_lstm = AirQualityLSTMDataset(X=X_val_seq, y=y_val_seq, input_window=lstm_window_size, horizon=HORIZON)
    training_loader_lstm = DataLoader(dataset=training_data_lstm, batch_size=16, shuffle=True)
    val_loader_lstm = DataLoader(dataset=val_data_lstm, batch_size=16, shuffle=False)

    history = train_lstm(
        model=lstm_model,
        train_loader=training_loader_lstm,
        val_loader=val_loader_lstm,
        n_epochs=100,
        lr=best_lstm_params["lr"],
        patience=10,
        recall=False
    )
else:
    try:
        with open("results/lstm_best_params.json", "r") as f:
            best_lstm_params = json.load(f)
            print("Loaded Optuna best parameters for LSTM")
    except FileNotFoundError:
        print("No Optuna parameters found, using default LSTM architecture")
        best_lstm_params = {"hidden_size": 128, "num_layers": 2, "dropout": 0.4, "lr": 1e-4, "window_size": 48}
        
    lstm_model = LSTMForecaster(
        n_features=X_train_seq.shape[1],
        hidden_size=best_lstm_params["hidden_size"],
        num_layers=best_lstm_params["num_layers"],
        horizon=HORIZON,
        dropout=best_lstm_params["dropout"],
    )
    
    lstm_window_size = best_lstm_params.get("window_size", 48)
    training_data_lstm = AirQualityLSTMDataset(X=X_train_seq, y=y_train_seq, input_window=lstm_window_size, horizon=HORIZON)
    val_data_lstm = AirQualityLSTMDataset(X=X_val_seq, y=y_val_seq, input_window=lstm_window_size, horizon=HORIZON)
    training_loader_lstm = DataLoader(dataset=training_data_lstm, batch_size=16, shuffle=True)
    val_loader_lstm = DataLoader(dataset=val_data_lstm, batch_size=16, shuffle=False)

    history = train_lstm(
        model=lstm_model,
        train_loader=training_loader_lstm,
        val_loader=val_loader_lstm,
        n_epochs=100,
        lr=best_lstm_params["lr"],
        patience=10,
        recall=False # Force retraining for log transformation 
    )


## Transformer model
if TUNE_TRANSFORMER:
    best_trans_params = tune_transformer(
        X_train=X_train_seq,
        y_train=y_train_seq,
        X_val=X_val_seq,
        y_val=y_val_seq,
        n_features=X_train_seq.shape[1],
        horizon=HORIZON,
        n_trials=30
    )
    with open("results/transformer_best_params.json", "w") as f:
        json.dump(best_trans_params, f, indent=4)
    transformer_model = TransformerForecaster(
        n_features=X_train_seq.shape[1],
        d_model=best_trans_params["d_model"],
        nhead=best_trans_params["nhead"],
        num_layers=best_trans_params["num_layers"],
        dropout=best_trans_params["dropout"],
        horizon=HORIZON
    )
    trans_window_size = best_trans_params.get("window_size", 48)
    training_data_trans = AirQualityLSTMDataset(X=X_train_seq, y=y_train_seq, input_window=trans_window_size, horizon=HORIZON)
    val_data_trans = AirQualityLSTMDataset(X=X_val_seq, y=y_val_seq, input_window=trans_window_size, horizon=HORIZON)
    training_loader_trans = DataLoader(dataset=training_data_trans, batch_size=16, shuffle=True)
    val_loader_trans = DataLoader(dataset=val_data_trans, batch_size=16, shuffle=False)

    history = train_transformer(
        model=transformer_model,
        train_loader=training_loader_trans,
        val_loader=val_loader_trans,
        n_epochs=100,
        lr=best_trans_params["lr"],
        patience=10,
        recall=False
    )
else:
    try:
        with open("results/transformer_best_params.json", "r") as f:
            best_trans_params = json.load(f)
            print("Loaded Optuna best parameters for Transformer")
    except FileNotFoundError:
        print("No Optuna parameters found, using default Transformer architecture")
        best_trans_params = {"d_model": 128, "nhead": 8, "num_layers": 2, "dropout": 0.3, "lr": 1e-5, "window_size": 48}
        
    transformer_model = TransformerForecaster(
        n_features=X_train_seq.shape[1],
        d_model=best_trans_params["d_model"],
        nhead=best_trans_params["nhead"],
        num_layers=best_trans_params["num_layers"],
        dropout=best_trans_params["dropout"],
        horizon=HORIZON
    )
    
    trans_window_size = best_trans_params.get("window_size", 48)
    training_data_trans = AirQualityLSTMDataset(X=X_train_seq, y=y_train_seq, input_window=trans_window_size, horizon=HORIZON)
    val_data_trans = AirQualityLSTMDataset(X=X_val_seq, y=y_val_seq, input_window=trans_window_size, horizon=HORIZON)
    training_loader_trans = DataLoader(dataset=training_data_trans, batch_size=16, shuffle=True)
    val_loader_trans = DataLoader(dataset=val_data_trans, batch_size=16, shuffle=False)

    history = train_transformer(
        model=transformer_model,
        train_loader=training_loader_trans,
        val_loader=val_loader_trans,
        n_epochs=100,
        lr=best_trans_params["lr"],
        patience=10,
        recall=False # Force retraining for log transformation 
    )


############################### QUICK MODEL EVALUATION ###############################

print("=========================================")
print("Starting model evaluation...")
print("=========================================")


metrics = {}

def compute_and_save(y_true, y_pred, model_name: str, scaler=None, is_log_target=False) -> dict:
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

    if is_log_target:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    np.save(f"results/predictions/{model_name}_y_true.npy", y_true)
    np.save(f"results/predictions/{model_name}_y_pred.npy", y_pred)
    return {"RMSE": rmse, "MAE": mae}


### Random Forest evaluation
y_pred_rf = rf_model.predict(X_test_rf)
metrics["random_forest"] = compute_and_save(y_test_rf, y_pred_rf, "rf", target_scaler_rf)

print("Generating Random Forest Feature Importance plot...")
import matplotlib.pyplot as plt
feature_names = scaler_rf.feature_names_in_
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 20
top_indices = indices[:top_n]

plt.figure(figsize=(12, 8))
plt.title(f"Random Forest - Top {top_n} Feature Importances", fontsize=16, fontweight="bold")
bars = plt.bar(range(top_n), importances[top_indices], align="center", color="#2ecc71", edgecolor="black", alpha=0.8)
plt.xticks(range(top_n), [feature_names[i] for i in top_indices], rotation=45, ha="right", fontsize=11)
plt.ylabel("Importance Score", fontsize=12, fontweight="bold")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.002, round(yval, 3), ha='center', va='bottom', fontsize=9, rotation=45)

plt.tight_layout()
plt.grid(axis='y', alpha=0.3)
os.makedirs("results/figures", exist_ok=True)
plt.savefig("results/figures/rf_feature_importance.png")
plt.close()
print("Saved Random Forest Feature Importance plot to results/figures/rf_feature_importance.png")



### Naive Baseline evaluation
y_true_naive = test_df[horizon_columns].values
y_pred_naive = np.tile(test_df[["PM2.5"]].values, (1, HORIZON))
metrics["naive"] = compute_and_save(y_true_naive, y_pred_naive, "naive", scaler=None)


### LSTM evaluation
lstm_window_size = best_lstm_params.get("window_size", 48)
test_data_lstm = AirQualityLSTMDataset(X=X_test_seq, y=y_test_seq, input_window=lstm_window_size, horizon=HORIZON)
test_loader_lstm = DataLoader(dataset=test_data_lstm, batch_size=16, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
lstm_model.eval()
preds, trues = [], []

with torch.no_grad():
    for x_batch, y_batch in test_loader_lstm:
        preds.append(lstm_model(x_batch).numpy())
        trues.append(y_batch.numpy())

y_pred_lstm = np.concatenate(preds)
y_true_lstm = np.concatenate(trues)
metrics["lstm"] = compute_and_save(y_true_lstm, y_pred_lstm, "lstm", target_scaler_seq, is_log_target=True)
    

### Transformer evaluation
trans_window_size = best_trans_params.get("window_size", 48)
test_data_trans = AirQualityLSTMDataset(X=X_test_seq, y=y_test_seq, input_window=trans_window_size, horizon=HORIZON)
test_loader_trans = DataLoader(dataset=test_data_trans, batch_size=16, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
transformer_model.eval()
preds, trues = [], []

with torch.no_grad():
    for x_batch, y_batch in test_loader_trans:
        x_batch = x_batch.to(device)
        preds.append(transformer_model(x_batch).cpu().numpy())
        trues.append(y_batch.numpy())

y_pred_transformer = np.concatenate(preds)
y_true_transformer = np.concatenate(trues)
metrics["transformer"] = compute_and_save(y_true_transformer, y_pred_transformer, "transformer", target_scaler_seq, is_log_target=True)
    

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
