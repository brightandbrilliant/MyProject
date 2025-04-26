import argparse
import torch
from torch_geometric.loader import DataLoader

from Train.Train import Trainer
from Clients.Client import Client
from torch.utils.data import Dataset
from torch_geometric.data import Data

import random


# ===== 自定义假数据集 =====
class MyFakeDataset(Dataset):
    def __init__(self, num_graphs=5, feature_dim=16, num_users=10):
        self.num_graphs = num_graphs
        self.feature_dim = feature_dim
        self.num_users = num_users

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        num_nodes = random.randint(2, 5)
        x = torch.randn((num_nodes, self.feature_dim))
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        batch = torch.zeros(num_nodes, dtype=torch.long)

        user_ids = torch.randint(0, self.num_users, (num_nodes,))
        target_labels = torch.randint(0, self.num_users, (num_nodes,))

        data = Data(x=x, edge_index=edge_index, batch=batch)
        data.user_ids = user_ids
        data.target_labels = target_labels
        return data


# ===== 构建 Clients =====
def build_clients(n_clients=3, feature_dim=16, num_users=10):
    clients = {}
    for cid in range(n_clients):
        client = Client(
            node_feat_dim=feature_dim,
            node_hidden_dim=32,
            node_embed_dim=64,
            graph_hidden_dim=32,
            graph_style_dim=32,
            fusion_output_dim=64,
            node_num_layers=2,
            graph_num_layers=2,
            dropout=0.1,
            n_clients=n_clients - 1,
            n_users=num_users
        )
        client.optimizer = torch.optim.Adam(client.parameters(), lr=0.01)
        clients[cid] = client
    return clients


# ===== 主函数入口 =====
def main():
    parser = argparse.ArgumentParser(description="Federated Graph Learning Trainer")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n_clients', type=int, default=3)
    parser.add_argument('--feature_dim', type=int, default=16)
    parser.add_argument('--num_users', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_graphs', type=int, default=5)
    parser.add_argument('--local_steps', type=int, default=1)
    parser.add_argument('--total_rounds', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    clients = build_clients(
        n_clients=args.n_clients,
        feature_dim=args.feature_dim,
        num_users=args.num_users
    )

    train_loaders = {}
    for cid in range(args.n_clients):
        dataset = MyFakeDataset(
            num_graphs=args.num_graphs,
            feature_dim=args.feature_dim,
            num_users=args.num_users
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        train_loaders[cid] = loader

    trainer = Trainer(
        clients=clients,
        train_loaders=train_loaders,
        device=args.device,
        local_steps=args.local_steps,
        total_rounds=args.total_rounds,
        alpha=args.alpha
    )

    trainer.train()


if __name__ == "__main__":
    main()


