import torch
from PIL import Image

from rotate_captcha_crack.common import device
from rotate_captcha_crack.const import DEFAULT_CLS_NUM
from rotate_captcha_crack.model import RotNetR, WhereIsMyModel
from rotate_captcha_crack.utils import get_state_dict, process_captcha

def predict_rotation_angle(image_path, model_path, model_index=-1):
    """预测图像的旋转角度。

    参数:
        image_path (str): 图像文件路径。
        model_index (int, 可选): 使用的模型索引。默认为-1。

    返回值:
        float: 预测的旋转角度，单位为度。
    """
    with torch.no_grad():
        cls_num = DEFAULT_CLS_NUM
        model = RotNetR(cls_num=cls_num, train=False)
        model_path = model_path
        print(f"Use model: {model_path}")
        model.load_state_dict(get_state_dict(model_path))
        model = model.to(device=device)
        model.eval()

        img = Image.open(image_path)
        img_ts = process_captcha(img, target_size=224)
        img_ts = img_ts.to(device=device)

        predict = model.predict(img_ts)
        degree = predict * 360
        print(f"Predict degree: {degree:.4f}°")
        
        return degree

def rotate_and_save_image(image_path, degree, output_path="debug.jpg"):
    """Rotate an image by the given degree and save it.

    Args:
        image_path (str): Path to the input image.
        degree (float): Rotation angle in degrees.
        output_path (str, optional): Path to save the rotated image. Defaults to "debug.jpg".
    """
    img = Image.open(image_path)
    img = img.rotate(
        -degree, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255)
    )  # use neg degree to recover the img
    img = img.convert('RGB')
    img.save(output_path)

if __name__ == "__main__":

    image_path = r"images\10.png"
    model_path = r'models\RotNetR\250101_23_38_54_001\best.pth'
    degree = predict_rotation_angle(image_path, model_path)
    print(f"Rotate {degree}°")
    rotate_and_save_image(image_path, degree)
