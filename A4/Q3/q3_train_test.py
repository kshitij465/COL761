import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

class LinkPredictionModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(LinkPredictionModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers - 1)])

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x

def prepare_data(data):
    # Splitting positive and negative edges into train and test
    pos_train, pos_test = train_test_split(data.positive_edges.t(), test_size=0.2, random_state=42)
    neg_train, neg_test = train_test_split(data.negative_edges.t(), test_size=0.2, random_state=42)

    # Combine and create labels
    train_edges = torch.cat([pos_train, neg_train], dim=0).t()
    test_edges = torch.cat([pos_test, neg_test], dim=0).t()
    train_labels = torch.cat([torch.ones(pos_train.size(0)), torch.zeros(neg_train.size(0))])
    test_labels = torch.cat([torch.ones(pos_test.size(0)), torch.zeros(neg_test.size(0))])
    
    return train_edges, train_labels, test_edges, test_labels

def train(model, data, edge_index, labels, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    x = model(data.x, data.edge_index)
    preds = (x[edge_index[0]] * x[edge_index[1]]).sum(dim=1)
    loss = criterion(preds, labels)
    loss.backward()
    optimizer.step()
    return loss

def evaluate(model, data, edge_index, labels):
    model.eval()
    with torch.no_grad():
        x = model(data.x, data.edge_index)
        preds = (x[edge_index[0]] * x[edge_index[1]]).sum(dim=1)
        preds = torch.sigmoid(preds)
        print(preds)
        print(labels)
        auc = roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())
        precision, recall, f1, _ = precision_recall_fscore_support(labels.cpu().numpy(), preds.round().cpu().numpy(), average='binary')
        print(f"ROC AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

def main():
    dataset_path = "../A4_Datasets/LP_D2.pt"
    dataset = torch.load(dataset_path)
    num_layers = 2  # You can adjust the number of layers as needed
    model = LinkPredictionModel(dataset.num_node_features, 128, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_edges, train_labels, test_edges, test_labels = prepare_data(dataset)
    for epoch in range(100):
        loss = train(model, dataset, train_edges, train_labels, optimizer, criterion)
        print(f"Epoch {epoch + 1}: Loss {loss:.4f}")

    evaluate(model, dataset, test_edges, test_labels)

if __name__ == "__main__":
    main()
