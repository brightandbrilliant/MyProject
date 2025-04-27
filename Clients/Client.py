import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from Components.GraphEmbeddingGenerator import GraphEmbeddingGenerator
from Components.NodeEmbeddingGenerator import UserEmbeddingGenerator
from Components.EmbeddingAggregator import PersonalizedUserAggregator
from Components.Fusion import FusionModule


class Client(nn.Module):
    def __init__(self,
                 node_feat_dim,          # 节点的初始特征维度
                 node_hidden_dim,        # 节点编码器的中间层维度
                 node_embed_dim,         # 节点最终嵌入维度
                 graph_hidden_dim,       # 图编码器中间层维度
                 graph_style_dim,        # 图嵌入最终输出维度（风格维度）
                 fusion_output_dim,      # 融合后输出维度
                 node_num_layers,             # GNN 层数
                 graph_num_layers,          # GNN 层数
                 dropout,                # Dropout 比例
                 n_clients,              # 除了本地端以外的客户数量
                 n_users,                # 用户总数（即所有 user id 的最大值 + 1，用于分类预测）
                 ):
        super(Client, self).__init__()

        # 节点嵌入生成器
        self.node_encoder = UserEmbeddingGenerator(
            input_dim=node_feat_dim,
            hidden_dim=node_hidden_dim,
            embed_dim=node_embed_dim,
            num_layers=node_num_layers,
            dropout=dropout
        )

        # 图嵌入生成器
        self.graph_encoder = GraphEmbeddingGenerator(
            node_feat_dim=node_feat_dim,
            hidden_dim=graph_hidden_dim,
            style_dim=graph_style_dim,
            num_layers=graph_num_layers,
            dropout=dropout
        )

        # 用户嵌入的个性化聚合器
        self.attn_aggregator = PersonalizedUserAggregator(
            embed_dim=node_embed_dim,
            n_clients=n_clients
        )

        # 节点嵌入与图嵌入的融合模块
        self.fusion = FusionModule(
            node_dim=node_embed_dim,
            graph_dim=graph_style_dim,
            out_dim=fusion_output_dim
        )

        # 用于最终关注用户预测的分类器
        self.predictor = nn.Linear(fusion_output_dim, n_users)

        # 交叉熵损失函数（用于多分类）
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, data, user_ids, target_labels, alpha, external_node_embeds_dict=None):
        """
        前向传播函数

        Args:
            data: 包含图结构数据，具有属性 x（特征）, edge_index, batch 等
            user_ids: List[int]，data 中每个节点对应的用户 ID，用于匹配其他客户端的嵌入
            target_labels: Tensor[int]，目标标签（节点真实关注的用户 ID），shape: [batch_size]
            external_node_embeds_dict: Dict[int, List[Tensor]]，每个用户 ID 在其他客户端中的嵌入
            alpha: 个性化聚合加权系数

        Returns:
            loss: CrossEntropy 损失
            logits: [batch_size, n_users]，用于后续评估等
        """

        # Step 1: 本地节点嵌入生成
        local_node_embeds = self.node_encoder(data)

        # Step 2: 聚合外部嵌入（若有）
        if external_node_embeds_dict is not None:
            aligned_external_embeds = []
            for i, uid in enumerate(user_ids):
                if uid in external_node_embeds_dict:
                    aligned_external_embeds.append(external_node_embeds_dict[uid])
                else:
                    # 若没有外部信息则使用全零代替
                    zeros = [torch.zeros_like(local_node_embeds[i]) for _ in range(self.attn_aggregator.n_clients)]
                    aligned_external_embeds.append(zeros)

            # 对齐外部嵌入格式
            aligned_external_embeds = [torch.stack(embed_list, dim=0) for embed_list in aligned_external_embeds]

            # 聚合外部和本地嵌入
            personalized_node_embed = self.attn_aggregator(aligned_external_embeds, local_node_embeds, alpha)
        else:
            personalized_node_embed = local_node_embeds

        # Step 3: 图嵌入生成
        graph_embed = self.graph_encoder(data)  # [num_graphs, graph_style_dim]

        # Step 4: 融合节点与图嵌入
        fused_embed = self.fusion(personalized_node_embed, graph_embed, batch=data.batch)  # [batch_size, fusion_output_dim]

        # Step 5: 进行多分类预测，目标是预测“关注哪个用户”
        logits = self.predictor(fused_embed)  # [batch_size, n_users]

        # Step 6: 计算损失
        loss = self.loss_fn(logits, target_labels)

        return loss, logits


"""
def test_client_forward():
    # 模拟图参数
    num_nodes = 6
    node_feat_dim = 8

    # 构造简单图数据
    x = torch.randn(num_nodes, node_feat_dim)
    edge_index = torch.randint(0, num_nodes, (2, 12))  # 随机边
    batch = torch.zeros(num_nodes, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, batch=batch)

    # 构造 user_ids 和 target_labels
    user_ids = list(range(num_nodes))
    target_labels = torch.randint(0, 20, (num_nodes,))  # 假设总用户数为 20

    # 构造外部嵌入字典（模拟 n_clients 个客户端）
    n_clients = 3
    node_embed_dim = 12
    external_node_embeds_dict = {
        uid: [torch.randn(node_embed_dim) for _ in range(n_clients)]
        for uid in user_ids
    }

    # 初始化 Client 模块
    model = Client(
        node_feat_dim=node_feat_dim,
        node_hidden_dim=16,
        node_embed_dim=node_embed_dim,
        graph_hidden_dim=16,
        graph_style_dim=6,
        fusion_output_dim=18,
        node_num_layers=2,
        graph_num_layers=2,
        dropout=0.2,
        n_clients=n_clients,
        n_users=20,
    )

    # 模型前向传播
    loss, logits = model(data, user_ids, target_labels, 0.8, external_node_embeds_dict)

    # 输出测试结果
    print("Test passed.")
    print("Loss:", loss.item())
    print("Logits shape:", logits.shape)  # 应为 [batch_size (= num_nodes), n_users]


test_client_forward()
"""
