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
    
    metrics_plot_path = "results/figures/metrics/metrics_comparison.png"
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
    
    rf_plot_path = "results/figures/predictions/rf_predictions.png"
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
    
    lstm_plot_path = "results/figures/predictions/lstm_predictions.png"
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
    
    horizon_plot_path = "results/figures/predictions/lstm_horizons.png"
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
        
        transformer_plot_path = "results/figures/predictions/transformer_predictions.png"
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
        
        trans_horizon_plot_path = "results/figures/predictions/transformer_horizons.png"
        plt.savefig(trans_horizon_plot_path)
        plt.close()
        print(f"Saved Transformer horizons plot to {trans_horizon_plot_path}")

        # 7. Residual Analysis (Scatter plot of True vs Residuals for t+1)
        print("Generating Residual Analysis scatter plot...")
        plt.figure(figsize=(12, 8))
        
        # Load RF predictions
        rf_true_res = np.load("results/predictions/rf_y_true.npy")[:, 0]
        rf_pred_res = np.load("results/predictions/rf_y_pred.npy")[:, 0]
        rf_residuals = rf_pred_res - rf_true_res
        
        # Load Transformer predictions
        transformer_true_res = np.load("results/predictions/transformer_y_true.npy")[:, 0]
        transformer_pred_res = np.load("results/predictions/transformer_y_pred.npy")[:, 0]
        transformer_residuals = transformer_pred_res - transformer_true_res
        
        # Load Naive predictions
        naive_true_res = np.load("results/predictions/naive_y_true.npy")[:, 0]
        naive_pred_res = np.load("results/predictions/naive_y_pred.npy")[:, 0]
        naive_residuals = naive_pred_res - naive_true_res
        
        # Load LSTM predictions
        try:
            lstm_true_res = np.load("results/predictions/lstm_y_true.npy")[:, 0]
            lstm_pred_res = np.load("results/predictions/lstm_y_pred.npy")[:, 0]
            lstm_residuals = lstm_pred_res - lstm_true_res
        except FileNotFoundError:
            lstm_true_res = []
            lstm_residuals = []
        
        plt.figure(figsize=(14, 10))
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        axes = axes.flatten()
        
        # Plot Naive
        axes[0].scatter(naive_true_res, naive_residuals, alpha=0.4, label="Naive", color="#95a5a6", s=15, edgecolors='none')
        axes[0].axhline(0, color='black', linestyle='--', linewidth=2)
        axes[0].set_title("Naive")
        axes[0].grid(True, alpha=0.3)
        
        # Plot RF
        axes[1].scatter(rf_true_res, rf_residuals, alpha=0.4, label="Random Forest", color="#2ecc71", s=15, edgecolors='none')
        axes[1].axhline(0, color='black', linestyle='--', linewidth=2)
        axes[1].set_title("Random Forest")
        axes[1].grid(True, alpha=0.3)
        
        # Plot LSTM
        if len(lstm_residuals) > 0:
            axes[2].scatter(lstm_true_res, lstm_residuals, alpha=0.4, label="LSTM", color="#e74c3c", s=15, edgecolors='none')
        axes[2].axhline(0, color='black', linestyle='--', linewidth=2)
        axes[2].set_title("LSTM")
        axes[2].grid(True, alpha=0.3)
        
        # Plot Transformer
        axes[3].scatter(transformer_true_res, transformer_residuals, alpha=0.4, label="Transformer", color="#3498db", s=15, edgecolors='none')
        axes[3].axhline(0, color='black', linestyle='--', linewidth=2)
        axes[3].set_title("Transformer")
        axes[3].grid(True, alpha=0.3)
        
        fig.suptitle("Residual Analysis (1-hour ahead): Error vs True PM2.5 Value", fontsize=16, fontweight="bold")
        fig.text(0.5, 0.04, "True PM2.5 Value (µg/m³)", ha='center', fontsize=12, fontweight="bold")
        fig.text(0.04, 0.5, "Residual (Predicted - True)", va='center', rotation='vertical', fontsize=12, fontweight="bold")
        
        residual_plot_path = "results/figures/residuals/residuals_scatter.png"
        plt.savefig(residual_plot_path)
        plt.close()
        print(f"Saved Residual scatter plot to {residual_plot_path}")
        
        # 8. Residual Distribution (Histogram)
        print("Generating Residual Distribution histogram...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        axes = axes.flatten()
        
        axes[0].hist(naive_residuals, bins=50, alpha=0.6, color="#95a5a6", edgecolor='black', density=True)
        axes[0].axvline(0, color='black', linestyle='--', linewidth=2)
        axes[0].set_title("Naive")
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(rf_residuals, bins=50, alpha=0.6, color="#2ecc71", edgecolor='black', density=True)
        axes[1].axvline(0, color='black', linestyle='--', linewidth=2)
        axes[1].set_title("Random Forest")
        axes[1].grid(True, alpha=0.3)
        
        if len(lstm_residuals) > 0:
            axes[2].hist(lstm_residuals, bins=50, alpha=0.6, color="#e74c3c", edgecolor='black', density=True)
        axes[2].axvline(0, color='black', linestyle='--', linewidth=2)
        axes[2].set_title("LSTM")
        axes[2].grid(True, alpha=0.3)
        
        axes[3].hist(transformer_residuals, bins=50, alpha=0.6, color="#3498db", edgecolor='black', density=True)
        axes[3].axvline(0, color='black', linestyle='--', linewidth=2)
        axes[3].set_title("Transformer")
        axes[3].grid(True, alpha=0.3)
        
        fig.suptitle("Distribution of Residuals (1-hour ahead)", fontsize=16, fontweight="bold")
        fig.text(0.5, 0.04, "Residual (Predicted - True)", ha='center', fontsize=12, fontweight="bold")
        fig.text(0.04, 0.5, "Density", va='center', rotation='vertical', fontsize=12, fontweight="bold")
        
        hist_plot_path = "results/figures/residuals/residuals_histogram.png"
        plt.savefig(hist_plot_path)
        plt.close()
        print(f"Saved Residual histogram plot to {hist_plot_path}")

        # 10. Autocorrelation of Residuals
        print("Generating Residual Autocorrelation plot...")
        from statsmodels.graphics.tsaplots import plot_acf
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
        
        plot_acf(naive_residuals, lags=48, ax=axes[0], color="#95a5a6", title="Naive Residuals Autocorrelation")
        plot_acf(rf_residuals, lags=48, ax=axes[1], color="#2ecc71", title="Random Forest Residuals Autocorrelation")
        if len(lstm_residuals) > 0:
            plot_acf(lstm_residuals, lags=48, ax=axes[2], color="#e74c3c", title="LSTM Residuals Autocorrelation")
        plot_acf(transformer_residuals, lags=48, ax=axes[3], color="#3498db", title="Transformer Residuals Autocorrelation")
        
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("Autocorrelation")
        
        axes[3].set_xlabel("Lags (hours)")
        plt.tight_layout()
        
        acf_plot_path = "results/figures/residuals/residuals_autocorrelation.png"
        plt.savefig(acf_plot_path)
        plt.close()
        print(f"Saved Residual ACF plot to {acf_plot_path}")

    except FileNotFoundError:
        print("Transformer predictions not found. Skipping Transformer plots.")

    # 9. Naive Baseline Predictions
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
        
        naive_plot_path = "results/figures/predictions/naive_predictions.png"
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
