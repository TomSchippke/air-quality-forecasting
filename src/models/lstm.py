'''
This is a simple LSTM model for time series forecasting.
'''


import torch 
import torch.nn as nn 
import os
import matplotlib.pyplot as plt
import optuna
import optuna.visualization.matplotlib as optuna_vis
from torch.utils.data import DataLoader
from src.data import AirQualityLSTMDataset

class LSTMForecaster(nn.Module):

    def __init__(
        self, 
        n_features: int, 
        hidden_size: int = 64,
        num_layers: int = 2,
        horizon: int = 6,
        dropout: float = 0.2,
    ) -> None:
        """
        Initialize the LSTMForecaster.

        Args:
            n_features (int): Number of features.
            hidden_size (int): Size of the hidden state.
            num_layers (int): Number of LSTM layers.
            horizon (int): Number of timesteps to predict.
            dropout (float): Dropout rate.
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.fc = nn.Linear( # project the hidden state to the output dimension
            hidden_size, 
            horizon
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Compute the output by applying the LSTM algorithm to x.
        x is a torch tensor of shape (batch_size, input_window, n_features)
        where input_window is the number of timesteps we use to predict the output
        and n_features is the number of features we use to predict the output
        We have this bacth_size number of samples at a time, with each sample containing
        input_window timesteps and n_features features 
        

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_window, n_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, horizon).
        """
        
        _, (h_n, c_n) = self.lstm(x)

        last_h = h_n[-1] # we take the hidden state of the LAST layer 
        out = self.fc(last_h) # we apply the linear layer to the last hidden state
        
        return out

def train_lstm(
    model: nn.Module,
    train_loader,
    val_loader,
    n_epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    recall: bool = False,
    model_path: str = "results/models/lstm_best.pth",
    figure_path: str = "results/figures/lstm_loss_curve.png",
    trial: optuna.Trial = None
) -> dict: 
    """
    Train the LSTM model.

    Args:
        model (nn.Module): The LSTM model to train.
        train_loader: The training data loader.
        val_loader: The validation data loader.
        n_epochs (int): The number of epochs to train for.
        lr (float): The learning rate.
        patience (int): The patience for early stopping.
        device (str): The device to train on.

    Returns:
        dict[str, list[float]]: A dictionary containing the training and validation losses.
    """

    if recall and os.path.exists(model_path):
        print(f"=========================================")
        print(f"Model already existing found here : {model_path}...")
        print(f"=========================================")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        return {}

    print(f"=========================================")
    print(f"Starting LSTM training...")
    print(f"=========================================")

    model.to(device) # move the model to the device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() 
    
    best_val_loss = float('inf') # initialize the best validation loss
    epoch_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}
    best_state = None

    for epoch in range(n_epochs): 
        
        ### training loop 
        model.train()
        train_loss = 0 

        for x_batch, y_batch in train_loader:

            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad() # put the gradient to 0
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch) # compute the loss
            loss.backward() # compute the gradient
            optimizer.step() # update the parameters

            train_loss += loss.item() * x_batch.size(0) # loss weighted by batch size
        
        train_loss /= len(train_loader.dataset) # loss devided by the number of samples


        ### validation loop 
        model.eval() # don't forget this (avoid dropout and other training behaviors)
        val_loss = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:

                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * x_batch.size(0)
        
            val_loss /= len(val_loader.dataset)   


        ### logging and saving        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}")
        

        ### early stopping 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epoch_without_improvement = 0
            best_state = model.state_dict()
        else:
            epoch_without_improvement += 1
            if epoch_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(best_state)
    

    ### save model and figure
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Best model saved in : {model_path}")

    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("LSTM Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(figure_path)
    plt.close()
    print(f"Loss curve saved in : {figure_path}")

    print(f"=========================================")
    print(f"END of LSTM training.")
    print(f"=========================================")

    return history

def tune_lstm(
    X_train,
    y_train,
    X_val,
    y_val,
    n_features: int,
    horizon: int = 6,
    n_trials: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """
    Optuna hyperparameter tuning for LSTM model.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_val: Validation features.
        y_val: Validation targets.
        n_features (int): Number of features.
        horizon (int): Number of timesteps to predict.
        n_trials (int): Number of trials.
        device (str): The device to train on.

    Returns:
        dict: A dictionary containing the best hyperparameters.
    """
    print("=========================================")
    print("Starting Optuna Hyperparameter Tuning for LSTM...")
    print("=========================================")

    def objective(trial):
        """
        Objective function for Optuna hyperparameter tuning.

        Args:
            trial (optuna.Trial): The Optuna trial object.

        Returns:
            float: The best validation loss.
        """
        lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True) # Widened LR search
        hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128, 256, 512]) # Expanded capacity
        num_layers = trial.suggest_int("num_layers", 1, 5) # Expanded layers
        dropout = trial.suggest_float("dropout", 0.0, 0.6) # Expanded dropout
        window_size = trial.suggest_categorical("window_size", [12, 24, 36, 48]) # Tuned window size

        training_data_toch = AirQualityLSTMDataset(
            X=X_train,
            y=y_train,
            input_window=window_size,
            horizon=horizon
        )

        val_data_toch = AirQualityLSTMDataset(
            X=X_val,
            y=y_val,
            input_window=window_size,
            horizon=horizon
        )

        train_loader = DataLoader(dataset=training_data_toch, batch_size=16, shuffle=True)
        val_loader = DataLoader(dataset=val_data_toch, batch_size=16, shuffle=False)

        model = LSTMForecaster(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            horizon=horizon,
            dropout=dropout,
        )

        history = train_lstm(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=40, # Shorter epochs for faster tuning
            lr=lr,
            patience=5,
            device=device,
            recall=False,
            trial=trial
        )
        return min(history["val_loss"])

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=7, n_warmup_steps=10, interval_steps=1)
    )
    study.optimize(objective, n_trials=n_trials)

    print("Best hyperparameters found by Optuna: ", study.best_params)

    # Save visualizations
    import os
    os.makedirs("results/figures", exist_ok=True)
    
    optuna_vis.plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig("results/figures/lstm_optuna_history.png")
    plt.close()

    optuna_vis.plot_param_importances(study)
    plt.tight_layout()
    plt.savefig("results/figures/lstm_optuna_importances.png")
    plt.close()

    return study.best_params

        
    
    
    
    