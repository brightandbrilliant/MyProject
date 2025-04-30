import torch
from torch_geometric.loader import DataLoader
from Clients.Client import Client
from sklearn.metrics import precision_score, recall_score
from Parse.BlogCatalog_parse import read_data
from Parse.BlogCatalog_parse.BlogCatalog_build import preprocess_social_graph


# ==== 1. 加载模型函数 ====
def load_model(checkpoint_path, model_args, device):
    model = Client(**model_args)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)  # 注意这里，直接load checkpoint
    model.to(device)
    model.eval()
    return model


def evaluate(model, dataloader, device, threshold=0.5):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            node_embeds = node_embeds = model.node_encoder(batch)
            graph_embed = model.graph_encoder(batch)
            fused = model.fusion(node_embeds, graph_embed, batch.batch)
            logits = model.predictor(fused)

            preds = torch.sigmoid(logits) > threshold
            all_preds.append(preds.cpu())
            all_labels.append(batch.target_labels.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    precision = precision_score(labels, preds, average='micro', zero_division=0)
    recall = recall_score(labels, preds, average='micro', zero_division=0)
    return precision, recall


def main():
    device = torch.device("cpu")

    # 模型初始化参数（要与训练时一致）
    model_args = dict(
        node_feat_dim=39,
        node_hidden_dim=32,
        node_embed_dim=64,
        graph_hidden_dim=32,
        graph_style_dim=32,
        fusion_output_dim=64,
        node_num_layers=2,
        graph_num_layers=2,
        dropout=0.1,
        n_clients=2,
        n_users=10312
    )

    # 加载模型
    checkpoint_path = 'Checkpoints/client_0_round_30.pth'  # 根据实际路径调整
    model = load_model(checkpoint_path, model_args, device)

    BlogCatalog_edge_path = './Dataset/BlogCatalog/BlogCatalog-dataset/data/edges.csv'
    BlogCatalog_group_path = './Dataset/BlogCatalog/BlogCatalog-dataset/data/group-edges.csv'

    raw_users = read_data(BlogCatalog_edge_path, BlogCatalog_group_path)
    test_data = preprocess_social_graph(raw_users)  # 返回的是一个 Data 对象
    test_dataset = [test_data]  # 包装成 list
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    # 模型评估
    precision, recall = evaluate(model, test_loader, device)
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")


if __name__ == '__main__':
    main()
