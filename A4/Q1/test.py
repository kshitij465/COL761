import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
import torch.nn.functional as F
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

def load_model(model_path, input_dim, hidden_dim, output_dim, device):
    model = GCN(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    return model

def test(model, loader,output_path):
    model.eval()
    predictions = []
    with open(output_path, 'w') as f:
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out = model(data)
                for x in F.softmax(out,dim=1):
                    f.write(str(x[1].item()) + '\n')

def main(model_path, dataset_path, output_path):
    dataset = load_dataset(dataset_path)
    input_dim = dataset[0].x.shape[1]
    hidden_dim = 64
    output_dim = 2
    model = load_model(model_path, input_dim, hidden_dim, output_dim, device)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    test(model, test_loader,output_path)



if __name__ == "__main__":
    model_path = sys.argv[1]
    dataset_path = sys.argv[2]
    output_path = sys.argv[3]
    main(model_path, dataset_path, output_path)
