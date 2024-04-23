import torch
import argparse
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.data import Data
import torch.optim as optim

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

def prepare_data(data):
    edge_index = torch.cat([data.positive_edges, data.negative_edges], dim=1)
    labels = torch.cat([torch.ones(data.positive_edges.size(1)), torch.zeros(data.negative_edges.size(1))])
    return edge_index, labels

def train(model, data, edge_index, labels, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    x = model(data.x, data.edge_index)
    preds = (x[edge_index[0]] * x[edge_index[1]]).sum(dim=1)
    loss = criterion(preds, labels)
    loss.backward()
    optimizer.step()
    return loss

def main(args):
    # Load dataset
    dataset = torch.load(args.dataset_path)
    model = LinkPredictionModel(dataset.num_node_features, 64)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    criterion = torch.nn.BCEWithLogitsLoss()

    edge_index, labels = prepare_data(dataset)
    for epoch in range(100):  # You can adjust the number of epochs
        loss = train(model, dataset, edge_index, labels, optimizer, criterion)
        print(f"Epoch {epoch + 1}: Loss {loss.item():.4f}")

    # Save the model
    torch.save(model.state_dict(), args.model_output_path)
    print(f"Model saved to {args.model_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GNN for link prediction and save the model.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset file")
    parser.add_argument("model_output_path", type=str, help="Path to save the trained model")
    args = parser.parse_args()
    main(args)
