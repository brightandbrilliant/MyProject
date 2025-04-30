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




