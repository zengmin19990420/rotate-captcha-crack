import os
import cv2
import time
from 旋转识别 import predict_rotation_angle

def process_images(model_path, images_dir):
    # 获取images文件夹中的所有图片
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))
    
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        
        # 读取原始图片
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图片: {image_path}")
            continue
            
        # 获取旋转角度
        angle = predict_rotation_angle(image_path,model_path)
        # from 识别 import predict_rotation
        # angle = predict_rotation(model_path, image_path)
        print(f"图片: {image_path}, 旋转角度: {angle}°")
        # 旋转图片
        center = (img.shape[1] // 2, img.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, 360 - angle, 1)
        rotated_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
        
        # 创建一个水平拼接的图像
        combined_img = cv2.hconcat([img, rotated_img])
        
        # 显示拼接后的图像
        cv2.imshow('原图 vs 旋转后', combined_img)
        
        # 等待1秒
        key = cv2.waitKey(1000)
        if key == 27:  # ESC键退出
            break
    
    cv2.destroyAllWindows()

def main():
    # 配置参数
    model_path = r'models\RotNetR\250101_23_38_54_001\best.pth'
    # model_path = r'rotnetr.onnx'
    images_dir = 'images'
    
    # 处理所有图片
    process_images(model_path, images_dir)

if __name__ == "__main__":
    main()
    # from 识别 import predict_rotation
    # angle = predict_rotation(model_path, image_path)
