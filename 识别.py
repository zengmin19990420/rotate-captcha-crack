import onnxruntime
import numpy as np
from PIL import Image
import warnings

def preprocess_image(img_path):
    try:
        img = Image.open(img_path)
        # 调整尺寸并转换为RGB
        img = img.convert('RGB').resize((224, 224))
        # 归一化处理 (根据模型训练时的预处理方式)
        img_array = np.array(img).transpose(2, 0, 1).astype(np.float32)
        img_array = (img_array / 255.0 - 0.5) * 2.0  # 假设使用-1到1的归一化
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"图像预处理失败: {str(e)}")
        return None

def predict_rotation(model_path, img_path):
    try:
        # 初始化ONNX推理会话
        session = onnxruntime.InferenceSession(model_path)
        
        # 预处理图像
        input_tensor = preprocess_image(img_path)
        if input_tensor is None:
            return None
            
        # 获取输入输出名称
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # 执行推理
        result = session.run([output_name], {input_name: input_tensor})

        
        # 解析输出角度 (假设输出为弧度值)
        angle_rad = result[0][0][0]
        angle_deg = np.degrees(angle_rad) % 360
        return angle_deg
    except Exception as e:
        print(f"预测失败: {str(e)}")
        return None

def rotate_image(img_path, angle, direction='cw'):
    try:
        img = Image.open(img_path)
        # 根据方向参数调整旋转角度（默认逆时针）
        rotate_angle = -angle if direction == 'cw' else angle
        rotated_img = img.rotate(rotate_angle, expand=True, resample=Image.BICUBIC)
        bbox = rotated_img.getbbox()
        cropped_img = rotated_img.crop(bbox) if bbox else rotated_img
        
        # 保存原图副本并调整尺寸
        original_img = img.copy()
        original_resized = original_img.resize(cropped_img.size)
        
        # 转换为OpenCV格式
        original_cv = cv2.cvtColor(np.array(original_resized), cv2.COLOR_RGB2BGR)
        rotated_cv = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)
        
        # 水平拼接显示
        combined = cv2.hconcat([original_cv, rotated_cv])
        cv2.imshow('Original vs Rotated', combined)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
        return True
    except Exception as e:
        print(f"图片旋转失败: {str(e)}")
        return False

def single_detection(model_path, image_path):
    angle = predict_rotation(model_path, image_path)
    if angle is not None:
        print(f"预测旋转角度: {angle:.2f}°")
        if rotate_image(image_path, angle, direction='ccw'):
            print("已显示校正后图片")
    else:
        print("角度预测失败")

def batch_detection(model_path, image_dir):
    import os
    import re
    supported_formats = ('.png', '.jpg', '.jpeg')
    for filename in sorted(os.listdir(image_dir), key=lambda x: int(re.findall(r'\d+', x)[0])):
        if filename.lower().endswith(supported_formats):
            image_path = os.path.join(image_dir, filename)
            print(f"\n正在处理: {filename}")
            single_detection(model_path, image_path)

if __name__ == "__main__":
    # 检查依赖库是否安装
    try:
        import onnxruntime
        import numpy
        from PIL import Image
        import cv2
        import os
        import argparse
    except ImportError as e:
        print("缺少依赖库，请执行以下命令安装:")
        print("pip install onnxruntime numpy pillow opencv-python")
        exit(1)

    parser = argparse.ArgumentParser(description='旋转验证码识别')
    parser.add_argument('--mode', choices=['single', 'batch'], default=None, help='运行模式：single(单张) / batch(批量)')
    parser.add_argument('--model', default=None, help='模型文件路径')
    parser.add_argument('--image', default=None, help='单张图片路径')
    parser.add_argument('--dir', default=None, help='图片目录路径')
    args = parser.parse_args()

    # 新增配置层（变量模式）
    config = {
        'mode': 'batch',
        'model': 'rotnetr.onnx',
        'image': 'images/11.png',
        'dir': 'images'
    }
    
    # 合并配置（命令行参数优先）
    config['mode'] = args.mode if args.mode is not None else config['mode']
    config['model'] = args.model if args.model is not None else config['model']
    config['image'] = args.image if args.image is not None else config['image']
    config['dir'] = args.dir if args.dir is not None else config['dir']

    if config['mode'] == 'single':
        single_detection(config['model'], config['image'])
    else:
        print(f"\n开始批量处理目录: {config['dir']}")
        batch_detection(config['model'], config['dir'])
        print("\n批量处理完成")