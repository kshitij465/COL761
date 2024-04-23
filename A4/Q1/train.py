import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
import numpy as np
import sys
device='cpu'
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=5):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList([GCNConv(input_dim, hidden_dim)] + 
                                         [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 2)] + 
                                         [GCNConv(hidden_dim, hidden_dim)])
        self.bns = torch.nn.ModuleList([BatchNorm(hidden_dim) for _ in range(num_layers - 1)])
        self.out = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index.long(), data.batch
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.out(x)
        return F.log_softmax(x, dim=1)

def load_dataset(path):
    return torch.load(path)

def split_dataset(dataset, test_size=0.01):
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    split = int(np.floor(test_size * len(indices)))
    train_idx, test_idx = indices[split:], indices[:split]
    train_dataset = [dataset[i] for i in train_idx]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    return train_loader

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def main(dataset_path, model_save_path):
    dataset = load_dataset(dataset_path)
    input_dim = dataset[0].x.shape[1]
    hidden_dim = 64
    output_dim = 2
    model = GCN(input_dim, hidden_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)  # Example weights
    total_samples=31638+1232
    if(input_dim==9):
        weights =torch.tensor([total_samples / 31638, total_samples / 1232], dtype=torch.float32).to(device)
        print("ghdjdk")
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    train_loader = split_dataset(dataset)
    if(input_dim==9):
        for epoch in range(50):  # Adjust epochs if needed
            train_loss = train(model, train_loader, optimizer, criterion)
            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')
    else:
        for epoch in range(500):  # Adjust epochs if needed
            train_loss = train(model, train_loader, optimizer, criterion)
            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')

    torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":
    dataset_path = sys.argv[1]
    model_save_path = sys.argv[2]
    main(dataset_path, model_save_path)
