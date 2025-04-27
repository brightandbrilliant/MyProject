import argparse
import torch
from torch_geometric.loader import DataLoader

from Parse.BlogCatalog_build import preprocess_social_graph
from Parse.BlogCatalog_parse import read_data
from Train.Train import Trainer
from Clients.Client import Client
from torch_geometric.data import Data
import random
import os


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
    parser.add_argument('--feature_dim', type=int, default=39)
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
        # ✅ 正确加载 BlogCatalog 数据
        BlogCatalog_edge_path = './Dataset/BlogCatalog/BlogCatalog-dataset/data/edges.csv'
        BlogCatalog_group_path = './Dataset/BlogCatalog/BlogCatalog-dataset/data/group-edges.csv'

        raw_users = read_data(BlogCatalog_edge_path, BlogCatalog_group_path)  # 得到字典 {user_id: info}
        # ✅ 不需要再转成 list！直接传进去
        data = preprocess_social_graph(raw_users)

        # ✅ 小trick：加 batch 标签（虽然在 preprocess 已经加了，但这里保险一点）
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

        # ✅ DataLoader要求list形式
        dataset = [data]
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        train_loaders[cid] = loader

    trainer = Trainer(
        clients=clients,
        train_loaders=train_loaders,
        device=args.device,
        local_steps=args.local_steps,
        total_rounds=args.total_rounds,  # ✅ 之前写死100，这里改成用args
        alpha=args.alpha,
        save_every=5,
        checkpoint_dir='checkpoints'
    )

    trainer.train(resume_round=0, load_checkpoint=False)


if __name__ == "__main__":
    main()


