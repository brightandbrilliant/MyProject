import torch
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
    def __init__(self, clients: Dict[int, Client], train_loaders: Dict[int, DataLoader], device, local_steps=1, total_rounds=10, alpha=0.5):
        """
        :param clients: 所有 client 的实例，格式 {client_id: client_instance}
        :param train_loaders: 每个 client 对应的 DataLoader
        :param device: 训练设备
        :param local_steps: 本地训练的轮数
        :param total_rounds: 联邦训练总轮数
        :param alpha: 融合外部嵌入时的个性化加权系数
        """
        self.clients = clients
        self.train_loaders = train_loaders
        self.device = device
        self.local_steps = local_steps
        self.total_rounds = total_rounds
        self.alpha = alpha

        # 每个 client 的通信器
        self.communicators = {
            cid: P2PCommunicator(client_id=cid, total_clients=list(clients.keys()))
            for cid in clients
        }

        # 模拟网络缓冲区，每轮广播之后清空
        self.network = {cid: [] for cid in clients}

    def train(self):
        for round in range(self.total_rounds):
            print(f"\n=== Federated Round {round + 1} ===")

            # 每个 client 本地训练 local_steps 次
            for cid, client in self.clients.items():
                communicator = self.communicators[cid]
                dataloader = self.train_loaders[cid]

                for local_step in range(self.local_steps):
                    # times = 1
                    # times用来显示信息
                    for batch in dataloader:
                        batch = batch.to(self.device)

                        user_ids = batch.user_ids.to(self.device)
                        target_labels = batch.target_labels.to(self.device)

                        # 整合外部嵌入（来自其他 clients）
                        external_dict = communicator.organize_external_node_embeds(
                            received_packets=self.network[cid],
                            user_ids=user_ids,
                            embed_dim=client.node_encoder.embed_dim,
                            device=self.device
                        )
                        # print(external_dict)
                        # 前向传播 + 反向传播
                        loss, _ = client.forward(
                            data=batch,
                            user_ids=user_ids,
                            target_labels=target_labels,
                            alpha=self.alpha,
                            external_node_embeds_dict=external_dict
                        )

                        # print(f"The loss of the {times}th training of client {cid}: {loss}")
                        # times += 1
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

            # 清空模拟网络缓冲区，为下一轮广播做准备
            self.network = {cid: [] for cid in self.clients}


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

