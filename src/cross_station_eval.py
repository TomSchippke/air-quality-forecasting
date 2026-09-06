import json
import numpy as np
import pandas as pd
import joblib
import torch
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader

from src.data import (
    load_and_preprocess, 
    add_lag_features, 
    temporal_split, 
    split_X_y_seq, 
    split_X_y_rf,
    AirQualityLSTMDataset
)
from src.models.lstm import LSTMForecaster
from src.models.transformer import TransformerForecaster

############################### GLOBAL PARAMETERS ###############################
HORIZON = 6
INCLUDE_PM10 = False
FEATURE_TO_DROP = ["datetime", "station", "No", "day", "year"]

# Grouped by rurality
URBAN_STATIONS = ["Aotizhongxin", "Dongsi", "Tiantan", "Guanyuan", "Wanshouxigong", "Nongzhanguan"]
SUBURBAN_STATIONS = ["Wanliu", "Shunyi", "Changping", "Gucheng"]
RURAL_STATIONS = ["Dingling", "Huairou"]

STATIONS = URBAN_STATIONS + SUBURBAN_STATIONS + RURAL_STATIONS

def load_models_and_scalers(n_features_seq):
    """ 
    Loads the best models and scalers from training.
    Infers PM2.5 for every station using the trained models.

    Args:
        n_features_seq (int): Number of features for the LSTM and Transformer models.

    Returns:
        dict: Dictionary containing the best models and scalers.
    """
    # Load scalers
    scaler_rf = joblib.load("results/scalers/scaler_rf.joblib")
    scaler_seq = joblib.load("results/scalers/scaler_seq.joblib")
    target_scaler_rf = joblib.load("results/scalers/target_scaler_rf.joblib")
    target_scaler_seq = joblib.load("results/scalers/target_scaler_seq.joblib")
    
    # Load RF
    rf_model = joblib.load("results/models/random_forest_best.joblib")
    
    # Load LSTM
    try:
        with open("results/lstm_best_params.json", "r") as f:
            best_lstm_params = json.load(f)
    except FileNotFoundError:
        best_lstm_params = {"hidden_size": 128, "num_layers": 2, "dropout": 0.4, "window_size": 48}
        
    lstm_model = LSTMForecaster(
        n_features=n_features_seq,
        hidden_size=best_lstm_params["hidden_size"],
        num_layers=best_lstm_params["num_layers"],
        horizon=HORIZON,
        dropout=best_lstm_params["dropout"],
    )
    lstm_model.load_state_dict(torch.load("results/models/lstm_best.pth", map_location="cpu", weights_only=True))
    lstm_model.eval()
    
    # Load Transformer
    try:
        with open("results/transformer_best_params.json", "r") as f:
            best_trans_params = json.load(f)
    except FileNotFoundError:
        best_trans_params = {"d_model": 128, "nhead": 8, "num_layers": 2, "dropout": 0.3, "window_size": 48}
        
    transformer_model = TransformerForecaster(
        n_features=n_features_seq,
        d_model=best_trans_params["d_model"],
        nhead=best_trans_params["nhead"],
        num_layers=best_trans_params["num_layers"],
        dropout=best_trans_params["dropout"],
        horizon=HORIZON
    )
    transformer_model.load_state_dict(torch.load("results/models/transformer_best.pth", map_location="cpu", weights_only=True))
    transformer_model.eval()
    
    return scaler_rf, scaler_seq, target_scaler_rf, target_scaler_seq, rf_model, lstm_model, transformer_model, best_lstm_params, best_trans_params

def evaluate_station(station, scaler_rf, scaler_seq, target_scaler_rf, target_scaler_seq, rf_model, lstm_model, transformer_model, best_lstm_params, best_trans_params):
    print(f"Evaluating {station}...")
    df = load_and_preprocess(location=station)
    df = add_lag_features(df, target="PM2.5", lags=[1,2,3,6,12,24], horizon=HORIZON)
    
    _, _, test_df = temporal_split(df)
    test_df = test_df.dropna().reset_index(drop=True)
    
    lag_columns = [f"PM2.5_lag_{l}" for l in [1, 2, 3, 6, 12, 24]]
    horizon_columns = [f"PM2.5_t+{l}" for l in range(1, HORIZON + 1)]
    
    X_test_seq, y_test_seq = split_X_y_seq(test_df, to_drop=FEATURE_TO_DROP + lag_columns + horizon_columns, include_PM10=INCLUDE_PM10)
    X_test_rf, y_test_rf = split_X_y_rf(test_df, horizon_columns=horizon_columns, include_PM10=INCLUDE_PM10)
    
    # Transform using pre-fitted scalers
    X_test_rf = scaler_rf.transform(X_test_rf)
    X_test_seq = scaler_seq.transform(X_test_seq)
    
    # We do NOT transform y because we want to evaluate on real original values directly.
    # The models will output scaled values, so we will inverse transform the predictions instead.
    
    # NAIVE
    y_true_naive = test_df[horizon_columns].values
    y_pred_naive = np.tile(test_df[["PM2.5"]].values, (1, HORIZON))
    rmse_naive = float(np.sqrt(mean_squared_error(y_true_naive, y_pred_naive)))
    mae_naive = float(mean_absolute_error(y_true_naive, y_pred_naive))
    
    # RANDOM FOREST
    y_pred_rf = rf_model.predict(X_test_rf)
    if target_scaler_rf.n_features_in_ == 1:
        orig_shape = y_pred_rf.shape
        y_pred_rf = target_scaler_rf.inverse_transform(y_pred_rf.flatten().reshape(-1, 1)).reshape(orig_shape)
    else:
        y_pred_rf = target_scaler_rf.inverse_transform(y_pred_rf)
    rmse_rf = float(np.sqrt(mean_squared_error(y_test_rf, y_pred_rf)))
    mae_rf = float(mean_absolute_error(y_test_rf, y_pred_rf))
    
    # DEEP LEARNING (LSTM + Transformer)
    lstm_window_size = best_lstm_params.get("window_size", 48)
    test_data_lstm = AirQualityLSTMDataset(X=X_test_seq, y=y_test_seq, input_window=lstm_window_size, horizon=HORIZON)
    test_loader_lstm = DataLoader(dataset=test_data_lstm, batch_size=32, shuffle=False)
    
    preds_lstm, trues_lstm = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader_lstm:
            preds_lstm.append(lstm_model(x_batch).numpy())
            trues_lstm.append(y_batch.numpy())
            
    y_pred_lstm = np.concatenate(preds_lstm)
    y_true_lstm = np.concatenate(trues_lstm)

    trans_window_size = best_trans_params.get("window_size", 48)
    test_data_trans = AirQualityLSTMDataset(X=X_test_seq, y=y_test_seq, input_window=trans_window_size, horizon=HORIZON)
    test_loader_trans = DataLoader(dataset=test_data_trans, batch_size=32, shuffle=False)
    
    preds_transformer, trues_trans = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader_trans:
            preds_transformer.append(transformer_model(x_batch).numpy())
            trues_trans.append(y_batch.numpy())
            
    y_pred_transformer = np.concatenate(preds_transformer)
    y_true_trans = np.concatenate(trues_trans)
    
    # Inverse transform
    if target_scaler_seq.n_features_in_ == 1:
        orig_shape_l = y_pred_lstm.shape
        orig_shape_t = y_pred_transformer.shape
        y_pred_lstm = target_scaler_seq.inverse_transform(y_pred_lstm.flatten().reshape(-1, 1)).reshape(orig_shape_l)
        y_pred_transformer = target_scaler_seq.inverse_transform(y_pred_transformer.flatten().reshape(-1, 1)).reshape(orig_shape_t)
    else:
        y_pred_lstm = target_scaler_seq.inverse_transform(y_pred_lstm)
        y_pred_transformer = target_scaler_seq.inverse_transform(y_pred_transformer)
        
    # The models were trained on log1p(PM2.5), so we apply expm1 to get raw PM2.5 back
    y_pred_lstm = np.expm1(y_pred_lstm)
    y_pred_transformer = np.expm1(y_pred_transformer)
        
    rmse_lstm = float(np.sqrt(mean_squared_error(y_true_lstm, y_pred_lstm)))
    mae_lstm = float(mean_absolute_error(y_true_lstm, y_pred_lstm))
    
    rmse_trans = float(np.sqrt(mean_squared_error(y_true_trans, y_pred_transformer)))
    mae_trans = float(mean_absolute_error(y_true_trans, y_pred_transformer))
    
    # Save CSV of predictions and residuals (1-hour ahead)
    n_seq = min(len(y_true_lstm), len(y_true_trans))
    
    naive_p = y_pred_naive[-n_seq:, 0]
    rf_p = y_pred_rf[-n_seq:, 0]
    true_vals = y_true_lstm[-n_seq:, 0]
    lstm_p = y_pred_lstm[-n_seq:, 0]
    trans_p = y_pred_transformer[-n_seq:, 0]
    
    df_preds = pd.DataFrame({
        "True_PM25": true_vals,
        "Naive_Pred": naive_p,
        "Naive_Residual": naive_p - true_vals,
        "RF_Pred": rf_p,
        "RF_Residual": rf_p - true_vals,
        "LSTM_Pred": lstm_p,
        "LSTM_Residual": lstm_p - true_vals,
        "Transformer_Pred": trans_p,
        "Transformer_Residual": trans_p - true_vals,
    })
    
    os.makedirs("results/station_predictions", exist_ok=True)
    df_preds.to_csv(f"results/station_predictions/{station}_predictions.csv", index=False)
    
    return {
        "naive": {"RMSE": rmse_naive, "MAE": mae_naive},
        "rf": {"RMSE": rmse_rf, "MAE": mae_rf},
        "lstm": {"RMSE": rmse_lstm, "MAE": mae_lstm},
        "transformer": {"RMSE": rmse_trans, "MAE": mae_trans}
    }

def main():
    # Number of features must match what was saved.
    # Typically 29 features for sequence models without PM10.
    # We can infer it by loading a dummy station.
    dummy_df = load_and_preprocess("Aotizhongxin")
    dummy_df = add_lag_features(dummy_df, lags=[1,2,3,6,12,24], horizon=HORIZON)
    lag_columns = [f"PM2.5_lag_{l}" for l in [1, 2, 3, 6, 12, 24]]
    horizon_columns = [f"PM2.5_t+{l}" for l in range(1, HORIZON + 1)]
    X_dummy, _ = split_X_y_seq(dummy_df.dropna(), to_drop=FEATURE_TO_DROP + lag_columns + horizon_columns, include_PM10=INCLUDE_PM10)
    n_features_seq = X_dummy.shape[1]
    
    scaler_rf, scaler_seq, target_scaler_rf, target_scaler_seq, rf_model, lstm_model, transformer_model, best_lstm_params, best_trans_params = load_models_and_scalers(n_features_seq)
    
    results = {}
    for station in STATIONS:
        results[station] = evaluate_station(
            station, scaler_rf, scaler_seq, target_scaler_rf, target_scaler_seq, rf_model, lstm_model, transformer_model, best_lstm_params, best_trans_params
        )
        
    # Save results to JSON
    with open("results/cross_station_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nEvaluations complete! Generating plots...")
    
    # Plotting RMSE
    fig, ax = plt.subplots(figsize=(14, 8))
    
    models = ["naive", "rf", "lstm", "transformer"]
    colors = ["#95a5a6", "#2ecc71", "#e74c3c", "#3498db"]
    x = np.arange(len(STATIONS))
    width = 0.2
    
    for i, model in enumerate(models):
        rmse_vals = [results[s][model]["RMSE"] for s in STATIONS]
        ax.bar(x + i*width - 1.5*width, rmse_vals, width, label=model.upper(), color=colors[i], edgecolor='black')
        
    ax.set_ylabel("RMSE (PM2.5 µg/m³)", fontweight="bold")
    ax.set_title("Cross-Station Generalization by Rurality (Trained on Aotizhongxin)", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(STATIONS, rotation=45, ha='right', fontsize=11)
    
    # Draw separators for rurality
    urban_end = len(URBAN_STATIONS) - 0.5
    suburban_end = len(URBAN_STATIONS) + len(SUBURBAN_STATIONS) - 0.5
    
    ax.axvline(x=urban_end, color='black', linestyle='--', linewidth=1.5)
    ax.axvline(x=suburban_end, color='black', linestyle='--', linewidth=1.5)
    
    max_ylim = ax.get_ylim()[1]
    ax.text(urban_end / 2 - 0.5, max_ylim * 0.95, 'URBAIN', ha='center', va='top', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
    ax.text(urban_end + len(SUBURBAN_STATIONS) / 2, max_ylim * 0.95, 'SUBURBAN', ha='center', va='top', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
    ax.text(suburban_end + len(RURAL_STATIONS) / 2, max_ylim * 0.95, 'RURAL', ha='center', va='top', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.85))
    plt.tight_layout()
    plt.savefig("results/figures/metrics/cross_station_comparison.png")
    plt.close()
    
    print("Plot saved to results/figures/metrics/cross_station_comparison.png")
    
if __name__ == "__main__":
    main()
