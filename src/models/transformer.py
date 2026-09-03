'''
This is 
'''

import math 
import os
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):
    '''
    Positional encoding adds information about the position of each token in the sequence.
    This is done by adding a vector of sines and cosines to the input embeddings.
    The formula for positional encoding is:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    where pos is the position of the token in the sequence and i is the dimension of the vector.
    '''

    def __init__(self, d_model: int, max_len: int = 500): 
        """
        Initialize the positional encoding.

        Args:
            d_model (int): Dimension of the model.
            max_len (int): Maximum length of the sequence.
        """
        super().__init__()

        pe = torch.zeros(max_len, d_model) # 
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # register_buffer used to store a non trainable matrix 


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the positional encoding.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, d_model).
        """
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        horizon: int = 6,
        dropout: float = 0.2,
    ):
        """
        Initialize the TransformerForecast.

        Args:
            n_features (int): Number of features.
            d_model (int): Dimension of the model. Should be divisible by nhead.
            nhead (int): Number of heads.
            num_layers (int): Number of layers.
            dim_feedforward (int): Dimension of the feedforward network.
            horizon (int): Number of timesteps to predict.
            dropout (float): Dropout rate.
        """
        super().__init__()

        self.embedding = nn.Linear(n_features, d_model) # maps input features to d_model dimension
        self.pos_encoding = PositionalEncoding(d_model) # adds positional encoding to the input

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, # d_model / nhead dimension for each head
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True, # so inputs are in shape (batch, sequence, features) instead of (sequence, batch, features)
        )
        
        self.encoder = nn.TransformerEncoder( # no need to have a mask here bc transformer is causal
            encoder_layer, 
            num_layers=num_layers # number of stacked encoder layers
        )

        self.fc = nn.Linear(d_model, horizon) # output linear layer


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, input_window, n_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch, horizon).
        """
        x = self.embedding(x)          
        x = self.pos_encoding(x)
        x = self.encoder(x)           
        last_step = x[:, -1, :]        
        return self.fc(last_step)      



def train_transformer(
    model: nn.Module,
    train_loader,
    val_loader,
    n_epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    recall: bool = False,
    model_path: str = "results/models/transformer_best.pth",
    figure_path: str = "results/figures/transformer_loss_curve.png",
) -> dict:
    """
    Train the transformer model.

    Args:
        model (nn.Module): The transformer model to train.
        train_loader: The training data loader.
        val_loader: The validation data loader.
        n_epochs (int): The number of epochs to train for.
        lr (float): The learning rate.
        patience (int): The patience for early stopping.
        device (str): The device to train on.
        recall (bool): Whether to recall the model.
        model_path (str): The path to save the model.
        figure_path (str): The path to save the figure.

    Returns:
        dict[str, list[float]]: A dictionary containing the training and validation losses.
    """
    
    if recall and os.path.exists(model_path):
        print(f"=========================================")
        print(f"Model already existing found here: {model_path}...")
        print(f"=========================================")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        return {}

    print(f"=========================================")
    print("Starting Transformer training...")
    print(f"=========================================")

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    epoch_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}
    best_state = None

    for epoch in range(n_epochs):
        model.train() 
        train_loss = 0

        for x_batch, y_batch in train_loader:

            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch) 
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0) 
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0 

        with torch.no_grad(): 
            for x_batch, y_batch in val_loader:

                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}")

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

    os.makedirs(os.path.dirname(model_path), exist_ok=True) 
    torch.save(model.state_dict(), model_path) 

    os.makedirs(os.path.dirname(figure_path), exist_ok=True) 
    plt.figure(figsize=(10, 5)) 
    plt.plot(history["train_loss"], label="Train Loss") 
    plt.plot(history["val_loss"], label="Validation Loss") 
    plt.title("Transformer Training and Validation Loss") 
    plt.xlabel("Epoch") 
    plt.ylabel("MSE Loss") 
    plt.legend() 
    plt.grid(True) 
    plt.savefig(figure_path) 
    plt.close() 

    print(f"=========================================")
    print("Transformer training completed.")
    print(f"=========================================")

    return history 
            