import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data


class UserEmbeddingGenerator(nn.Module):
    '''
    用户图嵌入生成器：用于生成中间节点表示，供个性化推荐使用。
    '''
    def __init__(self, input_dim, hidden_dim, embed_dim, num_layers=2, dropout=0.3):
        '''
        :param input_dim: 输入特征维度
        :param hidden_dim: 中间隐藏层维度
        :param embed_dim: 输出嵌入维度（推荐使用的用户表示）
        :param num_layers: 图卷积层数
        :param dropout: Dropout比例
        '''
        super(UserEmbeddingGenerator, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.embed_dim = embed_dim

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, embed_dim))  # 最后一层输出目标维度

    def forward(self, data):
        '''
        :param data: 包含图结构的 Data 对象（需包含 x 和 edge_index）
        :return: 节点嵌入（如用户表示）
        '''
        x, edge_index = data.x, data.edge_index

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        return x  # 返回节点的最终嵌入


"""
def test_user_embedding_generator():
    input_dim = 8
    hidden_dim = 16
    embed_dim = 10
    num_nodes = 5
    num_edges = 6

    # 构造节点特征
    x = torch.randn(num_nodes, input_dim)

    # 构造边（edge_index：形状为 [2, num_edges]）
    # 示例图为：0-1, 1-2, 2-3, 3-4, 4-0, 1-3（无向图，每条边要写双向）
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 1, 1, 2, 3, 4, 0, 3],
        [1, 2, 3, 4, 0, 0, 3, 1, 2, 3, 4, 1]
    ], dtype=torch.long)

    # 构造 Data 对象
    data = Data(x=x, edge_index=edge_index)

    # 实例化模型
    model = UserEmbeddingGenerator(input_dim, hidden_dim, embed_dim)

    # 前向传播
    node_embeddings = model(data)
    print(node_embeddings)
    # 检查输出维度
    assert node_embeddings.shape == (num_nodes, embed_dim), f"嵌入 shape 错误，实际为 {node_embeddings.shape}"
    print("UserEmbeddingGenerator 测试通过！")


# 运行测试
test_user_embedding_generator()
"""
