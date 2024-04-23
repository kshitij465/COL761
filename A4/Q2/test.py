import torch
import torch.nn.functional as F
import argparse
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import pandas as pd
def create_masks(data):
    test_mask = data.y == -1
    train_val_mask = data.y != -1
    train_val_indices = train_val_mask.nonzero(as_tuple=False).squeeze()
    train_indices, val_indices = train_test_split(train_val_indices, test_size=0.2, random_state=42)
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
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
        # return F.log_softmax(x, dim=1)
        return F.softmax(x, dim=1)

def evaluate(model, data, output_file):
    model.eval()
    out = model(data)
    out1=out
    out_array = out1.detach().numpy()

    # Convert the NumPy array to a pandas DataFrame
    df = pd.DataFrame(out_array)

    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False,header=False)


def test_main():
    parser = argparse.ArgumentParser(description="Evaluate a GraphSAGE model on node classification.")
    parser.add_argument('model_path', type=str, help="Path to the trained model file.")
    parser.add_argument('dataset_path', type=str, help="Path to the dataset file.")
    parser.add_argument('output_path', type=str, help="Path to save the output labels.")
    args = parser.parse_args()

    device = torch.device('cpu')
    data = load_dataset(args.dataset_path)
    data.train_mask, data.val_mask, data.test_mask = create_masks(data)
    data = data.to(device)
    feature_size = data.num_node_features
    hidden_size = 32
    num_classes = torch.unique(data.y[data.y != -1]).size(0)
    model = GraphSAGE(feature_size, hidden_size, num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path))  # Load your trained model weights
    evaluate(model, data, args.output_path)


if __name__ == "__main__":
    test_main()
