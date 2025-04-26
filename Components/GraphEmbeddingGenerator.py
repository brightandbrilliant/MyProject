import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.data import Data


class GraphEmbeddingGenerator(nn.Module):
    '''
    用于提取每个 client（如一个 app）的图风格嵌入。
    图嵌入将作为全局语境参与用户推荐，体现 client 偏好（如虎扑偏体育，知乎偏知识）。
    '''
    def __init__(self, node_feat_dim, hidden_dim, style_dim, num_layers=2, dropout=0.3):
        '''
        参数说明：
        node_feat_dim : 节点特征维度（输入维度）
        hidden_dim    : 中间 GCN 层维度
        style_dim     : 输出图风格嵌入维度
        num_layers    : 图卷积层数
        dropout       : dropout 概率
        '''
        super(GraphEmbeddingGenerator, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # 多层 GCNConv
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_feat_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # 将图嵌入投影为指定维度的风格向量
        self.convs.append(GCNConv(hidden_dim, style_dim))

    def forward(self, data):
        '''
        输入：
        data.x         节点特征 (num_nodes, node_feat_dim)
        data.edge_index 边信息
        返回：
        graph_style    每个图的风格嵌入 (num_graphs, style_dim)
        '''
        x, edge_index,batch = data.x, data.edge_index,data.batch

        # 多层 GCN
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        # 图级聚合
        graph_embed = global_add_pool(x, batch)

        return graph_embed


"""
def test_graph_embedding_generator():
    node_feat_dim = 8
    hidden_dim = 16
    style_dim = 10
    num_layers = 3
    num_nodes = 20
    dropout = 0.2

    model = GraphEmbeddingGenerator(node_feat_dim, hidden_dim, style_dim, num_layers, dropout)
    model.train()  # 训练模式

    # 构造 synthetic 图数据
    # 20 个节点，每个节点有 8 维特征
    x = torch.randn(num_nodes, node_feat_dim)

    # 构造一些简单边
    # 比如构成一个环状图
    edge_index = torch.tensor([[i for i in range(num_nodes)],
                               [(i + 1) % num_nodes for i in range(num_nodes)]], dtype=torch.long)

    # 构造 torch_geometric 的 Data 对象
    data = Data(x=x, edge_index=edge_index)

    # ----------- 测试 1: 前向传播 -----------
    graph_embed = model(data)
    assert graph_embed.shape == (1, style_dim), f"图嵌入 shape 应该是 [1, {style_dim}]，但得到了 {graph_embed.shape}"
    print("前向传播通过，输出形状正确！")

    # ----------- 测试 2: 反向传播 -----------
    loss = graph_embed.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"参数 {name} 没有梯度，反向传播失败"
    print("反向传播通过，梯度存在！")

    # ----------- 测试 3: 多种大小图 -----------
    for test_nodes in [5, 10, 50]:
        x = torch.randn(test_nodes, node_feat_dim)
        edge_index = torch.tensor([[i for i in range(test_nodes)],
                                   [(i + 1) % test_nodes for i in range(test_nodes)]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        out = model(data)
        print(out)
        assert out.shape == (1, style_dim), f"大小为 {test_nodes} 的图输出 shape 异常: {out.shape}"
    print("多图规模测试通过！")

    print("所有测试全部通过！")


# 运行测试
test_graph_embedding_generator()
"""







