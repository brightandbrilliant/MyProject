import pandas as pd
from collections import defaultdict


def read_data(edge_path, group_edge_path):
    edge_df = pd.read_csv(edge_path, sep=',', header=None, names=['user', 'friend'])
    group_edge_df = pd.read_csv(group_edge_path, sep=',', header=None, names=['user', 'group'])
    # print(edge_df.head())
    # print(group_edge_df.head())

    # 保证类型一致
    edge_df['user'] = edge_df['user'].astype(int)
    edge_df['friend'] = edge_df['friend'].astype(int)
    group_edge_df['user'] = group_edge_df['user'].astype(int)
    group_edge_df['group'] = group_edge_df['group'].astype(int)

    # 构建用户信息字典
    user_dict = defaultdict(lambda: {"following": [], "groups": [], "followers": []})
    for idx, row in edge_df.iterrows():
        user = row['user']
        following = row['friend']
        user_dict[user]["following"].append(following)
        user_dict[following]["followers"].append(user)

        # 遍历 user-group 边，填充 groups 列表
    for idx, row in group_edge_df.iterrows():
        user = row['user']
        group = row['group']
        user_dict[user]["groups"].append(group)

    user_dict = dict(user_dict)
    return user_dict


BlogCatalog_edge_path = '../Dataset/BlogCatalog/BlogCatalog-dataset/data/edges.csv'
BlogCatalog_group_path = '../Dataset/BlogCatalog/BlogCatalog-dataset/data/group-edges.csv'
dict_ = read_data(BlogCatalog_edge_path, BlogCatalog_group_path)
for i, user_id in enumerate(dict_.keys()):
    print(f"User ID: {user_id}")
    print(dict_[user_id])
    if i >= 4:  # 打印前5个用户信息
        break

