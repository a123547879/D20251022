import os
from Calculation_of_Shape_Consistency_Score_plus_Information_Entropy import *

root = 'save_models'

model_name = []

for name in os.listdir(root):
    model_name.append(name)
    # print(model_name)

ghost_model_files = []
mobV2_model_files = []
mobV3_model_files = []
shufV2_model_files = []
resNet18_model_files = []

for name in model_name:
    for file in os.listdir(os.path.join(root, name)):
        if '.pth' in file:
            if 'ghost' in name:
                ghost_model_files.append(os.path.join(root, name, file))
            elif 'mobV2' in name:
                mobV2_model_files.append(os.path.join(root, name, file))
            elif 'mobV3' in name:
                mobV3_model_files.append(os.path.join(root, name, file))
            elif 'shufV2' in name:
                shufV2_model_files.append(os.path.join(root, name, file))
            elif 'resNet18' in name:
                resNet18_model_files.append(os.path.join(root, name, file))
            else:
                pass

print('ghost_model_files: ', len(ghost_model_files))
print('mobV2_model_files: ', len(mobV2_model_files))
print('mobV3_model_files: ', len(mobV3_model_files))
print('shufV2_model_files: ', len(shufV2_model_files))
print('resNet18_model_files: ', len(resNet18_model_files))

def get_result_data(model_files, MODEL_TYPE, TARGET_LAYER_NAME):
    DATA_ROOT = "C:/Users/Administrator/Desktop/datas/test_data_black"
    MASK_THRESHOLD = 0.5
    HEATMAP_THRESHOLD = 0.5
    BATCH_SIZE = 10
    DEVICE = "cuda"

    for MODEL_FILE in model_files:
        # print(MODEL_FILE)
        if MODEL_TYPE == 'ghost_paper' and 'paper' not in MODEL_FILE:
            continue
        if MODEL_TYPE == 'ghost' and 'paper' in MODEL_FILE:
            continue
        if 'se' in MODEL_FILE:
            continue
        if 'exp' in MODEL_FILE:
            continue

        # print(model_file)
        chrs = MODEL_FILE.split('\\')
        # print(chrs)
        s = chrs[-1].split('_')
        # print(s[-1])
        IN_CHANNELS = int(s[-3][0])
        SEED = int(s[-1].split('.')[0])
        # print(IN_CHANNELS)
        
        main(DATA_ROOT,
            MODEL_TYPE,
            MODEL_FILE,
            IN_CHANNELS,
            TARGET_LAYER_NAME,
            MASK_THRESHOLD,
            HEATMAP_THRESHOLD,
            BATCH_SIZE,
            SEED,
            DEVICE
            )

ghost_paper_model_layers = [
    # 'features.1.ghost1.primary_conv.2', # 浅层 低级特征
    # 'features.5.ghost2.primary_conv.2', # 中层 中级特征
    # 'features.10.ghost2.primary_conv.2' # 深层 高级特征
    'blocks.0.0.ghost1.primary_conv.0',
    'blocks.3.0.ghost1.primary_conv.0',
    'blocks.6.2.ghost1.primary_conv.0'
]

ghost_model_layers = [
    'features.1.ghost1.primary_conv.2', # 浅层 低级特征
    'features.5.ghost2.primary_conv.2', # 中层 中级特征
    'features.10.ghost2.primary_conv.2' # 深层 高级特征
    # 'blocks.0.0.ghost1.primary_conv.0',
    # 'blocks.3.0.ghost1.primary_conv.0',
    # 'blocks.6.2.ghost1.primary_conv.0'
]  

mobV3_model_layers = [
    'features.init_conv.2', # 浅层 低级特征
    'features.bottleneck_5.conv3.1', # 中层 中级特征
    'features.final_conv.2' # 深层 高级特征
]

mobV2_model_layers = [
    # 'stem_conv.2', # 浅层 低级特征
    # 'last_conv', # 中层 中级特征
    # 'last_conv.1' #  深层 高级特征
    'stem_conv.2',
    'layers.4.layers.0',
    'last_conv.1'
]

shufV2_model_layers = [
    'first_conv.2', # 浅层 低级特征
    'features.6.branch_main.7', # 中层 中级特征
    'conv_last.2' # 深层 高级特征
]

resnet18_model_layers = [
    'layer1.1.conv2', # 浅层 低级特征
    'layer3.1.conv2', # 中层 中级特征
    'layer4.1.conv2' # 深层 高级特征
]

MODEL_TYPES = ['mobV3', 'ghost_paper', 'resNet18', 'mobV2', 'shufV2', 'ghost']
# MODEL_PATH = "save_models/ghost/GhostNet_3C_MD_42.pth"
# MODEL_TYPE = MODEL_TYPES[1]
TARGET_LAYER_NAME = "layer4.1.conv2"

for MODEL_TYPE, model_files in zip(MODEL_TYPES, [mobV3_model_files, ghost_model_files, resNet18_model_files, mobV2_model_files, shufV2_model_files, ghost_model_files]):
    # print(MODEL_TYPE)
    if MODEL_TYPE == 'mobV3':
        target_layer_names = mobV3_model_layers
        # continue
    elif MODEL_TYPE == 'ghost_paper':
        target_layer_names = ghost_paper_model_layers
        # continue
    elif MODEL_TYPE == 'resNet18':
        target_layer_names = resnet18_model_layers
        # continue
    elif MODEL_TYPE == 'mobV2':
        target_layer_names = mobV2_model_layers
        # continue
    elif MODEL_TYPE == 'shufV2':
        target_layer_names = shufV2_model_layers
        # continue
    elif MODEL_TYPE == 'ghost':
        target_layer_names = ghost_model_layers
        continue
    else:
        raise ValueError("Invalid model type.")

    for layer_name in target_layer_names:
        print(f'model_type: {MODEL_TYPE}')
        TARGET_LAYER_NAME = layer_name

        get_result_data(model_files, MODEL_TYPE, TARGET_LAYER_NAME)
