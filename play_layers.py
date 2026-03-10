import torch
from torch import nn
from models.GhostNet_paper import GhostNet
from models.MobileNetV2 import MobileNetV2
from models.MobileNetV3 import MobileNetV3
from models.ShuffleNetV2 import ShuffleNetV2
from models.ResNet18 import resnet18


def get_model_layers(model):
    layers = []
    for name, module in model.named_modules():
        # if any(keyword in name.lower() for keyword in ['conv', 'features', 'classifier', 'fc', 'bottleneck', 'block', 'layers']):
        layers.append(name)
    return layers

ghost_layers = []
mobV2_layers = []
mobV3_layers = []
shuf_layers = []
resnet18_layers = []

ghost = GhostNet(in_ch= 1)
ghost.load_state_dict(torch.load('save_models/ghost/GhostNet_paper_1C_MD_42.pth'))
ghost.eval()
ghost_layers = get_model_layers(ghost)

mobV2 = MobileNetV2(in_ch=1)
mobV2.load_state_dict(torch.load('save_models/mobV2/MobileNetV2_1C_MD_42.pth'))
mobV2.eval()
mobV2_layers = get_model_layers(mobV2)
# print(mobV2_layers)

mobV3 = MobileNetV3(version= 'small', in_ch=1)
mobV3.load_state_dict(torch.load('save_models/mobV3/MobileNetV3_1C_MD_42.pth'))
mobV3.eval()
mobV3_layers = get_model_layers(mobV3)

shuf = ShuffleNetV2(in_ch=1)
shuf.load_state_dict(torch.load('save_models/shufV2/ShuffleNetV2_1C_MD_42.pth'))
shuf.eval()
shuf_layers = get_model_layers(shuf)

resnet18 = resnet18(in_ch=1)
resnet18.load_state_dict(torch.load('save_models/resNet18/resnet18_1C_MD_42.pth'))
resnet18.eval()
resnet18_layers = get_model_layers(resnet18)

length = max(len(ghost_layers), len(mobV2_layers), len(mobV3_layers), len(shuf_layers), len(resnet18_layers))

with open('models/model_layers.csv', 'a+') as f:
    for i in range(length):
        try:
            ghost_layer = ghost_layers[i]
        except IndexError:
            ghost_layer = ''

        try:
            mobV2_layer = mobV2_layers[i]
        except IndexError:
            mobV2_layer = ''
        
        try:
            mobV3_layer = mobV3_layers[i]
        except IndexError:
            mobV3_layer = ''
        
        try:
            shuf_layer = shuf_layers[i]
        except IndexError:
            shuf_layer = ''
        
        try:
            resnet18_layer = resnet18_layers[i]
        except IndexError:
            resnet18_layer = ''
        
        line = f'{ghost_layer},{mobV2_layer},{mobV3_layer},{shuf_layer},{resnet18_layer}\n'
        f.writelines(line)
