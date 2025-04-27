import torch
from torch_geometric.data import Data

def preprocess_social_graph(raw_users):
    print(f"预处理开始，总共有 {len(raw_users)} 个用户")

    all_group_ids = set()
    valid_users = []

    # 遍历 key-value
    for user_id, user_info in raw_users.items():
        if not isinstance(user_info, dict):
            continue
        if 'groups' not in user_info:
            continue
        valid_users.append((user_id, user_info))
        all_group_ids.update(user_info['groups'])

    print(f"有效用户数量：{len(valid_users)}")
    print(f"出现的群组数量：{len(all_group_ids)}")

    if len(valid_users) == 0:
        raise ValueError("没有找到有效用户，请检查 raw_users 数据结构")

    all_group_ids = sorted(list(all_group_ids))
    group_id_to_idx = {gid: idx for idx, gid in enumerate(all_group_ids)}

    user_id2idx = {user_id: idx for idx, (user_id, _) in enumerate(valid_users)}

    x_list = []
    edge_index_list = []
    user_ids = []
    target_labels = []

    for user_id, user_info in valid_users:
        group_feats = torch.zeros(len(all_group_ids))
        for gid in user_info.get('groups', []):
            if gid in group_id_to_idx:
                group_feats[group_id_to_idx[gid]] = 1.0
        x_list.append(group_feats)
        user_ids.append(user_id2idx[user_id])

        # 注意：因为你的parse没有 target_label，这里用 -1 占位
        target_labels.append(-1)

        for followee in user_info.get('following', []):
            if followee in user_id2idx:
                src = user_id2idx[user_id]
                dst = user_id2idx[followee]
                edge_index_list.append([src, dst])

    if len(x_list) == 0:
        raise ValueError("所有用户都无效，导致无法构建特征矩阵")

    x = torch.stack(x_list)
    if len(edge_index_list) > 0:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    user_ids = torch.tensor(user_ids, dtype=torch.long)
    target_labels = torch.tensor(target_labels, dtype=torch.long)
    batch = torch.zeros(x.size(0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, batch=batch)
    data.user_ids = user_ids
    data.target_labels = target_labels
    print(f"预处理完成：节点数 {x.size(0)}, 边数 {edge_index.size(1)}")
    return data

