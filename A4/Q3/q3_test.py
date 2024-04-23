import torch
import argparse
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
from torch_geometric.data import Data
import numpy as np

class LinkPredictionModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(LinkPredictionModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def load_model(model_path, num_features, hidden_channels):
    model = LinkPredictionModel(num_features, hidden_channels)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model

def evaluate(model, data, edge_index, labels):
    model.eval()
    with torch.no_grad():
        x = model(data.x, data.edge_index)
        preds = (x[edge_index[0]] * x[edge_index[1]]).sum(dim=1)
        preds = torch.sigmoid(preds)
        auc = roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())
        precision, recall, f1, _ = precision_recall_fscore_support(labels.cpu().numpy(), preds.round().cpu().numpy(), average='binary')
        cm = confusion_matrix(labels.cpu().numpy(), preds.round().cpu().numpy())
        print(f"ROC AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")

def test(model, data):
    model.eval()
    with torch.no_grad():
        x = model(data.x, data.edge_index)
        test_preds = (x[data.test_edges[0]] * x[data.test_edges[1]]).sum(dim=1)
        test_preds = torch.sigmoid(test_preds)
        return test_preds

def save_predictions(predictions, file_path, num_nodes,data):
    with open(file_path, 'w') as file:
        # Loop through the tensor elements
        for item in predictions:
            # Write each element to a separate line in the file
            file.write(f"{item.item()}\n")

def main(args):
    dataset = torch.load(args.dataset_path)
    model = load_model(args.model_path, dataset.num_node_features, 64)
    predictions = test(model, dataset)
    save_predictions(predictions, args.output_path, len(dataset.x),dataset)
    print(f"Predictions saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a GNN for link prediction.")
    parser.add_argument("model_path", type=str, help="Path to the trained model file")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset file")
    parser.add_argument("output_path", type=str, help="Path to save the output predictions")
    args = parser.parse_args()
    main(args)
