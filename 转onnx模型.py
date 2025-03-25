import torch
import torch.nn as nn
from rotate_captcha_crack.model.rotr import RotNetR

# 初始化模型
model = RotNetR()

# 加载预训练权重
checkpoint = torch.load('models/RotNetR/250101_23_38_54_001/best.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()

# 创建示例输入
input_shape = (1, 3, 224, 224)
dummy_input = torch.randn(input_shape)

# 导出ONNX模型
torch.onnx.export(
    model,
    dummy_input,
    'rotnetr.onnx',
    export_params=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("ONNX模型导出成功！")