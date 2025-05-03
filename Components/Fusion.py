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


class AttentionalFusion(nn.Module):
    def __init__(self, node_dim, graph_dim, out_dim, hidden_dim=128):
        super(AttentionalFusion, self).__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1)
        self.output_proj = nn.Linear(node_dim + graph_dim, out_dim)

    def forward(self, node_embed, graph_embed, batch=None):
        if graph_embed.dim() == 2 and node_embed.size(0) != graph_embed.size(0):
            # 将图嵌入扩展到每个节点
            graph_embed = graph_embed[batch]

        # 计算注意力分数
        node_proj = torch.tanh(self.node_proj(node_embed))        # [N, H]
        graph_proj = torch.tanh(self.graph_proj(graph_embed))     # [N, H]

        combined = node_proj + graph_proj                         # [N, H]
        attn_weights = torch.sigmoid(self.attn_score(combined))   # [N, 1]

        # 使用注意力加权融合
        fused = attn_weights * node_embed + (1 - attn_weights) * graph_embed  # [N, D]
        out = self.output_proj(torch.cat([fused, node_embed], dim=-1))        # 可选：也可拼接 graph_embed
        return out


