"""
AbuseRing Sentinel - GNN Comparison Baseline (Rung 6)
Standard 2-Layer Graph Neural Network (GCN / GraphSAGE) using PyTorch.
Operates on the temporal bipartite graph projection with exact structural feature inputs.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

class GraphSAGENet(nn.Module):
    """
    Standard 2-Layer GraphSAGE model with Mean Aggregation.
    h_i^(l+1) = ReLU(W_self * h_i^(l) + W_neigh * mean_{j in N(i)}(h_j^(l)))
    """
    def __init__(self, in_dim: int, hidden_dim: int = 32, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.fc_self1 = nn.Linear(in_dim, hidden_dim)
        self.fc_neigh1 = nn.Linear(in_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc_self2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_neigh2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # Layer 1
        neigh1 = torch.sparse.mm(adj_norm, x) if adj_norm.is_sparse else torch.matmul(adj_norm, x)
        h1 = F.relu(self.fc_self1(x) + self.fc_neigh1(neigh1))
        h1 = self.dropout1(h1)
        
        # Layer 2
        neigh2 = torch.sparse.mm(adj_norm, h1) if adj_norm.is_sparse else torch.matmul(adj_norm, h1)
        h2 = F.relu(self.fc_self2(h1) + self.fc_neigh2(neigh2))
        h2 = self.dropout2(h2)
        
        logits = self.classifier(h2)
        return logits

def build_adj_matrix(G, account_ids: list[str]) -> torch.Tensor:
    """
    Builds row-normalized adjacency matrix (D^-1 A) with self-loops for the account set.
    """
    n = len(account_ids)
    acc_to_idx = {acc: i for i, acc in enumerate(account_ids)}
    
    # Adjacency with self-loops
    adj = np.eye(n, dtype=np.float32)
    
    for i, u in enumerate(account_ids):
        if G.has_node(u):
            for v, edge_data in G[u].items():
                if v in acc_to_idx:
                    j = acc_to_idx[v]
                    weight = float(edge_data.get("weight", 1.0))
                    adj[i, j] += weight
                    adj[j, i] += weight
                    
    # Row normalize: D^-1 A
    row_sum = adj.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    adj_norm = adj / row_sum
    
    return torch.tensor(adj_norm, dtype=torch.float32)

class GNNClassifier:
    """
    Scikit-learn compatible wrapper for the 2-layer GraphSAGE classifier.
    """
    def __init__(self, hidden_dim: int = 32, lr: float = 0.01, weight_decay: float = 1e-4,
                 epochs: int = 150, dropout: float = 0.1, random_state: int = 42):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dropout = dropout
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.training_time_sec = 0.0
        self.history = []

    def fit(self, X: np.ndarray, y: np.ndarray, G, account_ids: list[str],
            X_val: np.ndarray = None, y_val: np.ndarray = None, G_val = None, val_account_ids: list[str] = None):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        t0 = time.perf_counter()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        adj_norm = build_adj_matrix(G, account_ids)
        
        if X_val is not None and y_val is not None and G_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            x_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)
            adj_val_norm = build_adj_matrix(G_val, val_account_ids)
        else:
            x_val_tensor, y_val_tensor, adj_val_norm = None, None, None

        # Compute balanced class weights matching LightGBM
        classes, counts = np.unique(y, return_counts=True)
        class_weights = len(y) / (len(classes) * counts)
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
        
        in_dim = X.shape[1]
        self.model = GraphSAGENet(in_dim=in_dim, hidden_dim=self.hidden_dim, num_classes=len(classes), dropout=self.dropout)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        
        self.history = []
        best_val_loss = float("inf")
        best_state = None
        
        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            logits = self.model(x_tensor, adj_norm)
            loss = criterion(logits, y_tensor)
            loss.backward()
            optimizer.step()
            
            val_loss = None
            if x_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(x_val_tensor, adj_val_norm)
                    val_loss = criterion(val_logits, y_val_tensor).item()
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            
            self.history.append({
                "epoch": epoch + 1,
                "train_loss": round(loss.item(), 5),
                "val_loss": round(val_loss, 5) if val_loss is not None else None
            })
            
        if best_state is not None:
            self.model.load_state_dict(best_state)
            
        self.training_time_sec = round(time.perf_counter() - t0, 3)
        return self

    def predict_proba(self, X: np.ndarray, G, account_ids: list[str]) -> np.ndarray:
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        adj_norm = build_adj_matrix(G, account_ids)
        
        with torch.no_grad():
            logits = self.model(x_tensor, adj_norm)
            probs = F.softmax(logits, dim=1).numpy()
        return probs

    def predict(self, X: np.ndarray, G, account_ids: list[str]) -> np.ndarray:
        probs = self.predict_proba(X, G, account_ids)
        return np.argmax(probs, axis=1)
