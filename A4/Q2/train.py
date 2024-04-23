import torch
import argparse
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

def create_masks(data):
    test_mask = data.y == -1
    train_val_mask = data.y != -1
    train_val_indices = train_val_mask.nonzero(as_tuple=False).squeeze()
    train_indices, val_indices = train_test_split(train_val_indices, test_size=0.1, random_state=42)
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    # train_mask[val_indices] = True  # This line seems redundant as it re-assigns `True` to `val_indices` in `train_mask`
    return train_mask, val_mask, test_mask

def calculate_class_weights(labels):
    class_counts = labels.bincount()
    class_weights = 1. / class_counts.float()
    class_weights[class_weights == float('inf')] = 0
    return class_weights

def load_dataset(path):
    return torch.load(path)

class GraphSAGE(torch.nn.Module):
    def __init__(self, feature_size, hidden_size, num_classes, num_layers=3):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(feature_size, hidden_size))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_size, hidden_size))
        self.convs.append(SAGEConv(hidden_size, num_classes))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(x, dim=1)

def train(model, data, optimizer, criterion, mask):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data[mask]], data.y[data[mask]])
    loss.backward()
    optimizer.step()
    return loss.item()

def main():
    parser = argparse.ArgumentParser(description="Train a GraphSAGE model on node classification.")
    parser.add_argument('dataset_path', type=str, help="Path to the dataset file.")
    parser.add_argument('model_path', type=str, help="Path to save the trained model.")
    args = parser.parse_args()

    device = torch.device('cpu')
    data = load_dataset(args.dataset_path)
    data.train_mask, data.val_mask, data.test_mask = create_masks(data)
    class_weights = calculate_class_weights(data.y[data.y != -1]).to(device)
    data = data.to(device)
    class_weights = class_weights.to(device)
    feature_size = data.num_node_features
    hidden_size = 32
    num_classes = torch.unique(data.y[data.y != -1]).size(0)
    model = GraphSAGE(feature_size, hidden_size, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    criterion_weighted = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    for epoch in range(1000):  # Adjust the number of epochs as needed
        criterion_to_use = criterion_weighted if epoch > 50 else criterion
        train_loss = train(model, data, optimizer, criterion_to_use, 'train_mask')
        print(f'Epoch {epoch+1}: Training loss {train_loss:.4f}')
    
    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")

if __name__ == "__main__":
    main()
