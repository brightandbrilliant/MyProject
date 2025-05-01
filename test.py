import torch
from torch_geometric.loader import DataLoader
from Clients.Client import Client
from sklearn.metrics import precision_score, recall_score
import numpy as np


def load_model(checkpoint_path, model_args, device):
    model = Client(**model_args)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def evaluate(model, dataloader, device, threshold=0.5):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            node_embeds = model.node_encoder(batch)
            graph_embed = model.graph_encoder(batch)
            fused = model.fusion(node_embeds, graph_embed, batch.batch)
            logits = model.predictor(fused)

            print(fused)
            print(logits)
            # 计算预测概率
            probs = torch.sigmoid(logits)

            # 获取最终的预测结果
            preds = probs > threshold
            all_preds.append(preds.cpu())
            all_labels.append(batch.target_labels.cpu())

    # 合并所有预测结果
    preds = torch.cat(all_preds, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    precision = precision_score(labels, preds, average='micro', zero_division=0)
    recall = recall_score(labels, preds, average='micro', zero_division=0)

    return precision, recall


def main():
    device = torch.device("cpu")

    # 加载 test 数据
    test_data = torch.load('./Parsed_dataset/BlogCatalog/client0_train.pt')
    test_dataset = [test_data]  # 包装成 list
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 自动读取 n_users（即目标标签维度）
    n_users = test_data.target_labels.shape[0]

    # 模型初始化参数（要与训练时一致）
    model_args = dict(
        node_feat_dim=39,
        node_hidden_dim=32,
        node_embed_dim=64,
        graph_hidden_dim=32,
        graph_style_dim=64,
        fusion_output_dim=128,
        node_num_layers=3,
        graph_num_layers=3,
        dropout=0.1,
        n_clients=3,
        n_users=n_users
    )

    for i in range(1, 20):
        # 加载模型
        checkpoint_path = f'Check1/client_0_round_{50*i}.pth'
        model = load_model(checkpoint_path, model_args, device)
        # 模型评估
        precision, recall = evaluate(model, test_loader, device)
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")


if __name__ == '__main__':
    main()

