import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

def train_neural_network(X_train, y_train, X_val, y_val, X_test, y_test, epochs=100, lr=0.001):
      
    X_train_t = torch.FloatTensor(X_train.copy())
    y_train_t = torch.FloatTensor(y_train.copy()).reshape(-1, 1)
    X_val_t = torch.FloatTensor(X_val.copy())
    y_val_t = torch.FloatTensor(y_val.copy()).reshape(-1, 1)
    X_test_t = torch.FloatTensor(X_test.copy())
    y_test_t = torch.FloatTensor(y_test.copy()).reshape(-1, 1)
    
    model = SimpleNN(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses, val_losses = [], []
    best_val_loss, patience, best_state = float('inf'), 0, None
    
    print("\n" + "*"*60)
    print("TRAINING NEURAL NETWORK" + "\n")
    
    for epoch in range(epochs):

        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_loss = criterion(val_out, y_val_t).item()
            val_losses.append(val_loss)
            val_acc = ((val_out >= 0.5) == y_val_t).float().mean().item()
        
        if val_loss < best_val_loss:
            best_val_loss, patience = val_loss, 0
            best_state = model.state_dict().copy()
        else:
            patience += 1
            if patience >= 10:
                print(f"Early stopped at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.3f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_out = model(X_test_t)
        test_acc = ((test_out >= 0.5) == y_test_t).float().mean().item()
    
    print(f"\nTest Accuracy: {test_acc:.4f}" + "\n")
    
    # Save loss curves as PNG
    os.makedirs('models', exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.axvline(np.argmin(val_losses), color='red', linestyle='--', label='Best')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('models/loss_curves_neural_network.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Loss curves saved to models/loss_curves_neural_network.png")
    
    return model, test_acc

def save_model(model, filepath='models/neural_network.pt'):
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Saved to {filepath}")


def compare_activations(X_train, y_train, X_val, y_val, epochs=100, lr=0.001):
    """Compare different activation functions (ReLU, Leaky ReLU, ELU, Tanh)"""
    
    activations = {
        'ReLU': nn.ReLU(),
        'Leaky ReLU': nn.LeakyReLU(0.2),
        'ELU': nn.ELU(),
        'Tanh': nn.Tanh()
    }
    
    X_train_t = torch.FloatTensor(X_train.copy())
    y_train_t = torch.FloatTensor(y_train.copy()).reshape(-1, 1)
    X_val_t = torch.FloatTensor(X_val.copy())
    y_val_t = torch.FloatTensor(y_val.copy()).reshape(-1, 1)
    
    results = []
    
    print("\n" + "*"*60)
    print("ACTIVATION FUNCTION COMPARISON" + "\n")
    
    for act_name, activation in activations.items():
        model = nn.Sequential(
            nn.Linear(X_train.shape[1], 64),
            activation,
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            activation,
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X_train_t), y_train_t)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_t), y_val_t).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
        
        results.append({"Activation": act_name, "Best Val Loss": round(best_val_loss, 4)})
    
    print(pd.DataFrame(results).to_string(index=False))
    print("\n" + "*"*60 + "\n")