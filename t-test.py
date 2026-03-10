import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置中文字体
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 假设已经加载了各模型的准确率数据
# 这里用模拟数据展示，您需要用实际数据替换

# 模拟数据 - 请用您的实际数据替换
models_data = {
    'ResNet18_1ch': [0.9088, 0.9064, 0.9367, 0.9096, 0.9263],  # 5个种子的准确率
    'MobileNetV2_1ch': [0.8809, 0.9422, 0.9235, 0.9406, 0.9019],
    'MobileNetV3_1ch': [0.8785, 0.8227, 0.8478, 0.8598, 0.8737],
    'ShuffleNetV2_1ch': [0.8151, 0.8470, 0.9227, 0.8737, 0.8227],
    'GhostNet_1ch': [0.6968,0.8331, 0.7908, 0.8339,0.9108]
}

# 进行t检验比较
def perform_ttests(models_data, baseline='ResNet18_1ch'):
    """对每个模型与基准模型进行t检验"""
    ttest_results = []
    baseline_data = models_data[baseline]
    
    for model, data in models_data.items():
        if model != baseline:
            # 独立样本t检验
            t_stat, p_value = stats.ttest_ind(baseline_data, data)
            
            # 计算效应量 (Cohen's d)
            mean_diff = np.mean(baseline_data) - np.mean(data)
            pooled_std = np.sqrt((np.std(baseline_data, ddof=1)**2 + np.std(data, ddof=1)**2) / 2)
            cohens_d = mean_diff / pooled_std
            
            ttest_results.append({
                'Comparison': f'{baseline} vs {model}',
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05
            })
    
    return pd.DataFrame(ttest_results)

# 执行t检验
ttest_df = perform_ttests(models_data)

print("T检验结果:")
print(ttest_df.round(4))

# 绘制统计图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 图1: 各模型准确率分布（箱线图+散点）
model_names = list(models_data.keys())
accuracy_data = [models_data[name] for name in model_names]

# 箱线图
box_plot = ax1.boxplot(accuracy_data, labels=model_names, patch_artist=True)
# 散点图（显示每个数据点）
for i, data in enumerate(accuracy_data, 1):
    x = np.random.normal(i, 0.04, size=len(data))
    ax1.scatter(x, data, alpha=0.6, color='red', s=30)

ax1.set_title('各模型在五个随机种子下的准确率分布')
ax1.set_ylabel('准确率')
ax1.set_xlabel('模型配置')
ax1.grid(True, alpha=0.3)

# 图2: t检验结果可视化
comparisons = ttest_df['Comparison']
p_values = ttest_df['p_value']
cohens_d = ttest_df['cohens_d']

x_pos = np.arange(len(comparisons))
colors = ['red' if p < 0.05 else 'blue' for p in p_values]

bars = ax2.bar(x_pos, -np.log10(p_values), color=colors, alpha=0.7)
ax2.set_title('统计显著性比较 (-log10(p-value))')
ax2.set_ylabel('-log10(p-value)')
ax2.set_xlabel('模型比较')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([comp.split(' vs ')[1] for comp in comparisons], rotation=45)

# 添加显著性标注
for i, (bar, p_val, d_val) in enumerate(zip(bars, p_values, cohens_d)):
    height = bar.get_height()
    if p_val < 0.001:
        sig_text = '***'
    elif p_val < 0.01:
        sig_text = '**'
    elif p_val < 0.05:
        sig_text = '*'
    else:
        sig_text = 'n.s.'
    
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{sig_text}\nd={d_val:.2f}', ha='center', va='bottom')

ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05阈值')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('各模型单通道输入t检验.png', dpi=300)
plt.show()

# 输出统计摘要
print("\n统计摘要:")
for model, data in models_data.items():
    print(f"{model}: 均值={np.mean(data):.4f}, 标准差={np.std(data, ddof=1):.4f}, "
          f"95%置信区间=[{np.mean(data)-1.96*np.std(data, ddof=1)/np.sqrt(len(data)):.4f}, "
          f"{np.mean(data)+1.96*np.std(data, ddof=1)/np.sqrt(len(data)):.4f}]")
