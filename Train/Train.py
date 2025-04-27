import torch
import os
from torch_geometric.loader import DataLoader
from Clients.Client import Client
from Communication.P2P import P2PCommunicator
from typing import Dict

from torch_geometric.data import Data
from torch.utils.data import Dataset
import random


def to_device(data_dict, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}


class Trainer:
    def __init__(self, clients: Dict[int, Client], train_loaders: Dict[int, DataLoader], device, local_steps=1, total_rounds=10, alpha=0.5, save_every=1, checkpoint_dir='checkpoints', clean_old=True):
        """
        :param clients: 所有 client 的实例，格式 {client_id: client_instance}
        :param train_loaders: 每个 client 对应的 DataLoader
        :param device: 训练设备
        :param local_steps: 本地训练的轮数
        :param total_rounds: 联邦训练总轮数
        :param alpha: 融合外部嵌入时的个性化加权系数
        :param save_every: 每训练多少轮保存一次模型
        :param checkpoint_dir: 保存模型的文件夹
        :param clean_old: 是否在保存新 checkpoint 前清理旧的
        """
        self.clients = clients
        self.train_loaders = train_loaders
        self.device = device
        self.local_steps = local_steps
        self.total_rounds = total_rounds
        self.alpha = alpha
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir
        self.clean_old = clean_old

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.communicators = {
            cid: P2PCommunicator(client_id=cid, total_clients=list(clients.keys()))
            for cid in clients
        }

        self.network = {cid: [] for cid in clients}

    def train(self, resume_round=0, load_checkpoint=False):
        """
        :param resume_round: 从哪一轮开始训练
        :param load_checkpoint: 是否加载已有 checkpoint 继续训练
        """
        if load_checkpoint and resume_round > 0:
            self.load_checkpoint(resume_round)
            print(f"[Trainer] Loaded checkpoint from round {resume_round}.")

        for round in range(resume_round, self.total_rounds):
            print(f"\n=== Federated Round {round + 1} ===")

            for cid, client in self.clients.items():
                communicator = self.communicators[cid]
                dataloader = self.train_loaders[cid]

                for local_step in range(self.local_steps):
                    for batch in dataloader:
                        batch = batch.to(self.device)
                        user_ids = batch.user_ids.to(self.device)
                        target_labels = batch.target_labels.to(self.device)

                        external_dict = communicator.organize_external_node_embeds(
                            received_packets=self.network[cid],
                            user_ids=user_ids,
                            embed_dim=client.node_encoder.embed_dim,
                            device=self.device
                        )

                        loss, _ = client.forward(
                            data=batch,
                            user_ids=user_ids,
                            target_labels=target_labels,
                            alpha=self.alpha,
                            external_node_embeds_dict=external_dict
                        )

                        loss.backward()
                        client.optimizer.step()
                        client.optimizer.zero_grad()

                # 广播本地用户嵌入
                with torch.no_grad():
                    for batch in dataloader:
                        batch = batch.to(self.device)
                        user_ids = batch.user_ids.to(self.device)
                        local_user_embeds = client.node_encoder(batch)
                        package = communicator.pack_user_embeddings(user_ids, local_user_embeds)
                        communicator.send_to_all_peers(package, self.network)

            # 清空模拟网络缓冲区
            self.network = {cid: [] for cid in self.clients}

            # 保存 checkpoint
            if (round + 1) % self.save_every == 0:
                self.save_checkpoint(round + 1)

    def save_checkpoint(self, round):
        """保存所有 clients 的模型参数"""
        if self.clean_old:
            self._clear_checkpoints()

        for cid, client in self.clients.items():
            save_path = os.path.join(self.checkpoint_dir, f'client_{cid}_round_{round}.pth')
            torch.save(client.state_dict(), save_path)
        print(f"[Checkpoint] Saved models at round {round}.")

    def load_checkpoint(self, round):
        """加载所有 clients 的模型参数"""
        for cid, client in self.clients.items():
            load_path = os.path.join(self.checkpoint_dir, f'client_{cid}_round_{round}.pth')
            if os.path.exists(load_path):
                state_dict = torch.load(load_path, map_location=self.device)
                client.load_state_dict(state_dict)
                print(f"[Checkpoint] Loaded client {cid} model from round {round}.")
            else:
                print(f"[Checkpoint] Warning: No checkpoint found for client {cid} at round {round}.")

    def _clear_checkpoints(self):
        """清理 checkpoint 目录下所有旧的 .pth 文件"""
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('.pth'):
                filepath = os.path.join(self.checkpoint_dir, filename)
                os.remove(filepath)
        print(f"[Checkpoint] Cleared old checkpoints.")


"""
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


def build_clients(n_clients=2, feature_dim=16, num_users=10):
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


def test_my_trainer():
    torch.manual_seed(0)
    n_clients = 3
    feature_dim = 16
    num_users = 10
    batch_size = 1

    clients = build_clients(n_clients=n_clients, feature_dim=feature_dim, num_users=num_users)

    train_loaders = {}
    for cid in range(n_clients):
        dataset = MyFakeDataset(num_graphs=5, feature_dim=feature_dim, num_users=num_users)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # 使用 PyG 的 DataLoader
        train_loaders[cid] = loader

    trainer = Trainer(
        clients=clients,
        train_loaders=train_loaders,
        device='cpu',
        local_steps=3,
        total_rounds=5,
        alpha=0.5
    )

    trainer.train()


if __name__ == "__main__":
    test_my_trainer()
"""

