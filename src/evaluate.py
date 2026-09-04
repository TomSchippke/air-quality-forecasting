import json
import numpy as np
import matplotlib.pyplot as plt
import os


def main():

    print("=========================================")
    print("Starting figure generation from metrics...")
    print("=========================================")

    os.makedirs("results/figures", exist_ok=True)
    
    print("Loading metrics from results/metrics.json...")
    with open("results/metrics.json", "r") as f:
        metrics = json.load(f)
        
    # Sort models by RMSE (lowest to highest, meaning best to worst)
    sorted_metrics = sorted(metrics.items(), key=lambda item: item[1]["RMSE"])
    models = [item[0].upper() for item in sorted_metrics]
    rmse_vals = [item[1]["RMSE"] for item in sorted_metrics]
    mae_vals = [item[1]["MAE"] for item in sorted_metrics]
    
    x = np.arange(len(models))
    width = 0.35

    # 1. Bar chart comparing RMSE and MAE
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Using a modern palette
    color_rmse = '#1E88E5' # vibrant blue
    color_mae = '#FFC107'  # vibrant amber
    
    rects1 = ax.bar(x - width/2, rmse_vals, width, label='RMSE', color=color_rmse, edgecolor='black', linewidth=0.5)
    rects2 = ax.bar(x + width/2, mae_vals, width, label='MAE', color=color_mae, edgecolor='black', linewidth=0.5)

    # Add numeric labels on top of the bars
    ax.bar_label(rects1, padding=3, fmt='%.1f', fontsize=10, fontweight='bold')
    ax.bar_label(rects2, padding=3, fmt='%.1f', fontsize=10, fontweight='bold')
    
    # Ensure there is enough space on the y-axis for the labels
    max_val = max(max(rmse_vals), max(mae_vals))
    ax.set_ylim(0, max_val * 1.15)

    ax.set_ylabel('Scores (PM2.5 µg/m³)', fontsize=12, fontweight='bold')
    ax.set_title('Comparison of RMSE and MAE between Models (Best to Worst)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    metrics_plot_path = "results/figures/metrics_comparison.png"
    plt.savefig(metrics_plot_path)
    plt.close()
    print(f"Saved metrics comparison to {metrics_plot_path}")

    # 2. Random Forest Predictions
    print("Generating Random Forest prediction plot...")
    rf_true = np.load("results/predictions/rf_y_true.npy")
    rf_pred = np.load("results/predictions/rf_y_pred.npy")
    
    plt.figure(figsize=(14, 6))
    # We plot the first 200 points to keep the chart readable
    rf_true_curve = rf_true[:200, 0] if len(rf_true.shape) > 1 else rf_true[:200]
    rf_pred_curve = rf_pred[:200, 0] if len(rf_pred.shape) > 1 else rf_pred[:200]
    
    plt.plot(rf_true_curve, label="True PM2.5", color="blue", linewidth=1.5)
    plt.plot(rf_pred_curve, label="Predicted PM2.5", color="red", linestyle="--", linewidth=1.5)
    plt.title("Random Forest: Real vs Predicted (First 200 hours)")
    plt.xlabel("Hours")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend()
    plt.grid(True)
    
    rf_plot_path = "results/figures/rf_predictions.png"
    plt.savefig(rf_plot_path)
    plt.close()
    print(f"Saved Random Forest predictions plot to {rf_plot_path}")
    
    # 3. LSTM Predictions
    print("Generating LSTM prediction plot...")
    lstm_true = np.load("results/predictions/lstm_y_true.npy")
    lstm_pred = np.load("results/predictions/lstm_y_pred.npy")
    
    plt.figure(figsize=(14, 6))
    # The LSTM outputs an array of shape (N, 6) because horizon=6.
    # We will plot the 1-hour ahead prediction (index 0) for the first 200 windows.
    plt.plot(lstm_true[:200, 0], label="True PM2.5 (t+1)", color="blue", linewidth=1.5)
    plt.plot(lstm_pred[:200, 0], label="Predicted PM2.5 (t+1)", color="red", linestyle="--", linewidth=1.5)
    plt.title("LSTM: Real vs Predicted 1-hour ahead (First 200 windows)")
    plt.xlabel("Time Window")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend()
    plt.grid(True)
    
    lstm_plot_path = "results/figures/lstm_predictions.png"
    plt.savefig(lstm_plot_path)
    plt.close()
    print(f"Saved LSTM predictions plot to {lstm_plot_path}")

    # 4. Plot evolution of prediction horizons
    print("Generating LSTM prediction horizons plot...")
    plt.figure(figsize=(14, 8))
    
    # We want to plot the True values and the predictions for horizons 1 to 6
    # To align them on the same target time T:
    # lstm_true[T, 0] is the true value at target time T.
    # lstm_pred[T - h, h] is the prediction for T made h+1 hours ago.
    
    T_start = 150
    T_end = 350
    time_axis = np.arange(T_start, T_end)
    
    true_curve = lstm_true[T_start:T_end, 0]
    
    plt.plot(time_axis, true_curve, label="True PM2.5", color="black", linewidth=2.5)
    
    colors = ['#ffb3b3', '#ff8080', '#ff4d4d', '#e60000', '#b30000', '#800000']
    
    for h in range(6):
        curve_h = []
        for T in time_axis:
            if T - h >= 0:
                curve_h.append(lstm_pred[T - h, h])
            else:
                curve_h.append(np.nan)
        plt.plot(time_axis, curve_h, label=f"Prediction at t-{h+1}h", color=colors[h], alpha=0.9, linewidth=1.5)
        
    plt.title("LSTM: Impact of the prediction horizon (Hours 150 to 350)")
    plt.xlabel("Time (Target Time T)")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    horizon_plot_path = "results/figures/lstm_horizons.png"
    plt.savefig(horizon_plot_path)
    plt.close()
    print(f"Saved LSTM horizons plot to {horizon_plot_path}")

    # 5. Transformer Predictions
    print("Generating Transformer prediction plot...")
    try:
        transformer_true = np.load("results/predictions/transformer_y_true.npy")
        transformer_pred = np.load("results/predictions/transformer_y_pred.npy")
        
        plt.figure(figsize=(14, 6))
        plt.plot(transformer_true[:200, 0], label="True PM2.5 (t+1)", color="blue", linewidth=1.5)
        plt.plot(transformer_pred[:200, 0], label="Predicted PM2.5 (t+1)", color="red", linestyle="--", linewidth=1.5)
        plt.title("Transformer: Real vs Predicted 1-hour ahead (First 200 windows)")
        plt.xlabel("Time Window")
        plt.ylabel("PM2.5 (µg/m³)")
        plt.legend()
        plt.grid(True)
        
        transformer_plot_path = "results/figures/transformer_predictions.png"
        plt.savefig(transformer_plot_path)
        plt.close()
        print(f"Saved Transformer predictions plot to {transformer_plot_path}")

        # 6. Plot evolution of prediction horizons for Transformer
        print("Generating Transformer prediction horizons plot...")
        plt.figure(figsize=(14, 8))
        
        T_start = 200
        T_end = 300
        time_axis = np.arange(T_start, T_end)
        
        true_curve = transformer_true[T_start:T_end, 0]
        
        plt.plot(time_axis, true_curve, label="True PM2.5", color="black", linewidth=2.5)
        
        colors = ['#ffb3b3', '#ff8080', '#ff4d4d', '#e60000', '#b30000', '#800000']
        
        for h in range(6):
            curve_h = []
            for T in time_axis:
                if T - h >= 0:
                    curve_h.append(transformer_pred[T - h, h])
                else:
                    curve_h.append(np.nan)
            plt.plot(time_axis, curve_h, label=f"Prediction at t-{h+1}h", color=colors[h], alpha=0.9, linewidth=1.5)
            
        plt.title("Transformer: Impact of the prediction horizon (Hours 200 to 300)")
        plt.xlabel("Time (Target Time T)")
        plt.ylabel("PM2.5 (µg/m³)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        trans_horizon_plot_path = "results/figures/transformer_horizons.png"
        plt.savefig(trans_horizon_plot_path)
        plt.close()
        print(f"Saved Transformer horizons plot to {trans_horizon_plot_path}")
    except FileNotFoundError:
        print("Transformer predictions not found. Skipping Transformer plots.")

    # 7. Naive Baseline Predictions
    print("Generating Naive Baseline prediction plot...")
    try:
        naive_true = np.load("results/predictions/naive_y_true.npy")
        naive_pred = np.load("results/predictions/naive_y_pred.npy")
        
        plt.figure(figsize=(14, 6))
        plt.plot(naive_true[:200, 0], label="True PM2.5 (t+1)", color="blue", linewidth=1.5)
        plt.plot(naive_pred[:200, 0], label="Predicted PM2.5 (t+1)", color="red", linestyle="--", linewidth=1.5)
        plt.title("Naive Baseline: Real vs Predicted 1-hour ahead (First 200 hours)")
        plt.xlabel("Hours")
        plt.ylabel("PM2.5 (µg/m³)")
        plt.legend()
        plt.grid(True)
        
        naive_plot_path = "results/figures/naive_predictions.png"
        plt.savefig(naive_plot_path)
        plt.close()
        print(f"Saved Naive Baseline predictions plot to {naive_plot_path}")
    except FileNotFoundError:
        print("Naive predictions not found. Skipping Naive plots.")

    print("=========================================")
    print("End of figure generation.")
    print("=========================================")

if __name__ == "__main__":
    main()
