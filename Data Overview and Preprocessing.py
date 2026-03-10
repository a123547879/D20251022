import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置中文字体
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
data = pd.read_csv('data_csv/result.csv')

# 定义层级映射
layer_depth_mapping = {
    # MobileNetV3
    'features.init_conv.2': '浅层',
    'features.bottleneck_5.conv3.1': '中层', 
    'features.final_conv.2': '深层',
    # GhostNet
    'features.1.ghost1.primary_conv.2': '浅层',
    'features.5.ghost2.primary_conv.2': '中层',
    'features.10.ghost2.primary_conv.2': '深层',
    # ResNet18
    'layer1.1.conv2': '浅层',
    'layer3.1.conv2': '中层',
    'layer4.1.conv2': '深层',
    # MobileNetV2
    'stem_conv.2': '浅层',
    'last_conv': '中层',
    'last_conv.1': '深层',
    # ShuffleNetV2
    'first_conv.2': '浅层',
    'features.6.branch_main.7': '中层',
    'conv_last.2': '深层'
}
 
# 2.2 信息熵随网络深度的演变
# 添加层级信息
data['layer_depth'] = data['target_layer_name'].map(layer_depth_mapping)

print("数据概览:")
print(f"总样本数: {len(data)}")
print(f"模型类型: {data['model_type'].unique().tolist()}")
print(f"网络层级: {data['layer_depth'].unique().tolist()}")

# 按模型和层级分组统计
layer_iou_stats = data.groupby(['model_type', 'layer_depth'])['avg_iou'].agg([
    'mean', 'std', 'count'
]).round(4)

print("各模型在不同层级的形状一致性IoU:")
print(layer_iou_stats)

# 可视化
plt.figure(figsize=(12, 8))
sns.lineplot(data=data, x='layer_depth', y='avg_iou', hue='model_type', 
             marker='o', markersize=8, linewidth=2.5)
plt.title('形状一致性IoU随网络深度的演变')
plt.ylabel('平均IoU')
plt.xlabel('网络层级')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('data_csv/result_iou_layer_depth.png', dpi=300)
plt.show()


# 3.1 各模型的特征关注演化模式
# 计算每个模型从浅层到深层的IoU增长倍数
def calculate_iou_growth(data):
    growth_data = []
    for model in data['model_type'].unique():
        model_data = data[data['model_type'] == model]
        shallow_iou = model_data[model_data['layer_depth'] == '浅层']['avg_iou'].mean()
        deep_iou = model_data[model_data['layer_depth'] == '深层']['avg_iou'].mean()
        
        if shallow_iou > 0:  # 避免除零
            growth_ratio = deep_iou / shallow_iou
        else:
            growth_ratio = float('inf')
            
        growth_data.append({
            'model': model,
            'shallow_iou': shallow_iou,
            'deep_iou': deep_iou, 
            'growth_ratio': growth_ratio
        })
    
    return pd.DataFrame(growth_data)

growth_df = calculate_iou_growth(data)
print("形状一致性从浅层到深层的增长:")
print(growth_df.sort_values('growth_ratio', ascending=False))

# 3.2 关键发现总结
# 创建演化模式分类
evolution_patterns = {
    '渐进式聚焦': ['ResNet18', 'ShuffleNetV2'],  # IoU稳步提升
    '早期锁定': ['MobileNetV3'],  # 中层就达到较高IoU
    '持续分散': ['GhostNet'],     # 始终低IoU，关注点分散
    '早期高关注': ['MobileNetV2'] # 浅层就有高IoU，但深层反而下降
}

print("各模型的关注演化模式:")
for pattern, models in evolution_patterns.items():
    print(f"{pattern}: {', '.join(models)}")


# 分析不同结构在不同层级的效果
structure_analysis = data.groupby(['model_type', 'layer_depth']).agg({
    'avg_iou': 'mean',
    'avg_entropy': 'mean'
}).unstack()

print("结构组件在不同层级的有效性:")
print(structure_analysis.round(4))


# 将分层形状一致性与域外准确率关联
# 这是审稿人最可能问的问题："这些模式真的能预测性能吗？"

# 假设的准确率数据（用您的真实数据替换）
accuracy_data = {
    'mobV3': 0.8646,
    'ghost': 0.55, 
    'resNet18': 0.9176,
    'mobV2': 0.9076,
    'shufV2': 0.8562
}

# 分析深层IoU与准确率的关联
deep_layer_data = data[data['layer_depth'] == '深层']
deep_iou_vs_accuracy = []
for model in deep_layer_data['model_type'].unique():
    model_deep_iou = deep_layer_data[deep_layer_data['model_type'] == model]['avg_iou'].mean()
    accuracy = accuracy_data.get(model, 0)
    deep_iou_vs_accuracy.append({
        'model': model,
        'deep_iou': model_deep_iou,
        'accuracy': accuracy
    })

correlation_df = pd.DataFrame(deep_iou_vs_accuracy)
# 计算相关系数 皮尔逊积矩相关系数
correlation = np.corrcoef(correlation_df['deep_iou'], correlation_df['accuracy'])[0, 1]
print(f"深层形状一致性与域外准确率的相关系数: {correlation:.4f}")
