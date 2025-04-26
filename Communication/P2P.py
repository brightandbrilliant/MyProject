import torch
from typing import Dict, List


class P2PCommunicator:
    def __init__(self, client_id, total_clients):
        """
        :param client_id: 当前客户端编号
        :param total_clients: 所有客户端编号列表（例如 [0, 1, 2]）
        """
        self.client_id = client_id
        self.peers = [cid for cid in total_clients if cid != client_id]

    def pack_user_embeddings(self, user_ids, user_embeds):
        """
        将用户嵌入打包为 P2P 通信格式。

        Args:
            client_id (int): 当前客户端编号
            user_ids (Tensor): shape = [B]
            user_embeds (Tensor): shape = [B, D]

        Returns:
            dict: {
                "client_id": int,
                "client_node_embeddings": [(user_id, user_embedding), ...]
            }
        """
        packed = {
            "client_id": self.client_id,
            "client_node_embeddings": [
                (int(uid), embed.detach().cpu()) for uid, embed in zip(user_ids, user_embeds)
            ]
        }
        return packed

    def send_to_all_peers(self, package: Dict, network: Dict[int, List[Dict]]):
        """
        模拟将打包数据广播给其他所有 clients
        :param package: 打包数据
        :param network: 全局模拟网络环境（dict：client_id -> list of received messages）
        """
        for peer_id in self.peers:
            network[peer_id].append(package)

    def organize_external_node_embeds(self, received_packets, user_ids, embed_dim, device):
        """
        将所有收到的客户端广播包，按 user_id 聚合为 external_node_embeds_dict。

        Args:
            received_packets (List[Dict]): 来自其他客户端的广播包，每个格式为：
                {
                    "client_id": int,
                    "client_node_embeddings": List[Tuple[int, Tensor]]
                }
            user_ids (Tensor): 当前 batch 的 user_id，shape = [B]
            embed_dim (int): 嵌入维度
            device: torch.device

        Returns:
            Dict[int, List[Tensor]]: external_node_embeds_dict[user_id] = [client1_embed, ..., clientN_embed]
        """
        n_clients = len(received_packets)
        external_node_embeds_dict = {}

        for uid in user_ids.tolist():
            external_node_embeds_dict[uid] = []

            for packet in received_packets:
                if packet["client_id"] == self.client_id:
                    continue  # 跳过自己

                # 构建 user_id -> embed 的映射
                embed_map = dict(packet["client_node_embeddings"])

                if uid in embed_map:
                    external_node_embeds_dict[uid].append(embed_map[uid].to(device))
                else:
                    external_node_embeds_dict[uid].append(torch.zeros(embed_dim, device=device))  # 缺失填零

        return external_node_embeds_dict

"""
def test_p2p_with_global_node_embeddings():
    # 假设共有3个 client
    client_ids = [0, 1, 2]
    device = torch.device("cpu")
    embed_dim = 10
    all_user_ids = [100, 101, 102, 103, 104]  # 假设所有 client 都共享这5个节点ID
    user_id_tensor = torch.tensor(all_user_ids)

    # client 0 的 batch 用户
    batch_user_ids = torch.tensor([101, 103])

    # 每个 client 生成全局节点嵌入
    global_node_embeddings = {
        cid: torch.randn(len(all_user_ids), embed_dim) for cid in client_ids
    }

    # 初始化 communicator 和网络环境
    communicators = {
        cid: P2PCommunicator(client_id=cid, total_clients=client_ids)
        for cid in client_ids
    }
    network = {cid: [] for cid in client_ids}

    # 每个 client 对 batch_user_ids 中的节点提取嵌入并打包广播
    for cid in client_ids:
        # 获取该 client 的完整 embedding 表
        node_table = global_node_embeddings[cid]  # shape = [num_nodes, embed_dim]

        # 根据 user_id 顺序提取嵌入
        id_to_idx = {uid: i for i, uid in enumerate(all_user_ids)}
        embed_batch = torch.stack([node_table[id_to_idx[int(uid)]] for uid in batch_user_ids])

        # 打包并广播
        package = communicators[cid].pack_user_embeddings(batch_user_ids, embed_batch)
        communicators[cid].send_to_all_peers(package, network)

    # client 0 整理自己接收到的外部 embedding
    received_packets = network[0]
    external_embeds_dict = communicators[0].organize_external_node_embeds(
        received_packets, batch_user_ids, embed_dim, device
    )
    print(external_embeds_dict)

    # 打印验证结果
    print("==== External Embeddings for Client 0 ====")
    for uid in batch_user_ids.tolist():
        print(f"User ID {uid}:")
        for i, emb in enumerate(external_embeds_dict[uid]):
            print(f"  From client {communicators[0].peers[i]}: {emb.numpy()}")


# 运行测试函数
test_p2p_with_global_node_embeddings()
"""
