import torch
import torch.nn as nn


class FusionModule(nn.Module):
    def __init__(self, node_dim, graph_dim, out_dim):
        super(FusionModule, self).__init__()
        self.fc = nn.Linear(node_dim + graph_dim, out_dim)

    def forward(self, node_embed, graph_embed, batch=None):
        if len(graph_embed.shape) == 2 and node_embed.shape[0] != graph_embed.shape[0]:
            graph_embed = graph_embed[batch]  # 匹配每个节点所在图

        concat = torch.cat([node_embed, graph_embed], dim=-1)
        return self.fc(concat)

"""
def test_fusion_module():
    node_dim = 8
    graph_dim = 10
    out_dim = 18
    num_nodes = 5
    num_graphs = 2

    model = FusionModule(node_dim, graph_dim, out_dim)

    node_embed = torch.randn(num_nodes, node_dim)
    graph_embed = torch.randn(num_graphs, graph_dim)
    batch = torch.tensor([0, 0, 1, 1, 1])  # 节点分别属于图0、0、1、1、1

    output = model(node_embed, graph_embed, batch)
    print(output)
    assert output.shape == (num_nodes, out_dim), f"输出 shape 错误，得到 {output.shape}"

    print("✅ FusionModule 测试通过！")


# 运行测试
test_fusion_module()
"""


