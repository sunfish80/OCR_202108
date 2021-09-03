import numpy as np
import random
import cv2 as cv
from PIL import Image, ImageEnhance, ImageDraw


# # 定义相关utils

# * 函数 resize_img：调整图片大小
# * 函数 random_brightness：随机调整亮度，进行数据增强
# * 函数 random_contrast：随机调整对比度，进行数据增强
# * 函数 random_saturation：：随机调整饱和度，进行数据增强
# * 函数 random_hue：：随机调整色相，进行数据增强
# * 函数 distort_image：将上述数据增强手段整合，施加到训练样本上
# * 函数 rotate_image：随机旋转图片，进行数据增强
# * 函数 random_expand：随机改变图片大小，进行数据增强
# * 函数 preprocess：调用上述函数对训练样本进行数据增强及标准化

train_opt = {
    "mean_color": 127.0,
    "image_distort_strategy": {
        "expand_prob": 0.3,
        "expand_max_ratio": 2.0,
        "hue_prob": 0.5,
        "hue_delta": 48,
        "contrast_prob": 0.5,
        "contrast_delta": 0.3,
        "saturation_prob": 0.5,
        "saturation_delta": 0.3,
        "brightness_prob": 0.5,
        "brightness_delta": 0.3,
    }
}
def resize_img(img, input_size):
    target_size = input_size
    percent_h = float(target_size[1]) / img.size[1]
    percent_w = float(target_size[2]) / img.size[0]
    percent = min(percent_h, percent_w)
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    w_off = (target_size[2] - resized_width) / 2
    h_off = (target_size[1] - resized_height) / 2
    img = img.resize((resized_width, resized_height), Image.ANTIALIAS)
    array = np.ndarray((target_size[1], target_size[2], 3), np.uint8)
    array[:, :, 0] = 127
    array[:, :, 1] = 127
    array[:, :, 2] = 127
    ret = Image.fromarray(array)
    ret.paste(img, (np.random.randint(0, w_off + 1), int(h_off)))
    return ret


def random_brightness(img):
    prob = np.random.uniform(0, 1)
    if prob < train_opt['image_distort_strategy']['brightness_prob']:
        brightness_delta = train_opt['image_distort_strategy']['brightness_delta']
        delta = np.random.uniform(-brightness_delta, brightness_delta) + 1
        img = ImageEnhance.Brightness(img).enhance(delta)
    return img


def random_contrast(img):
    prob = np.random.uniform(0, 1)
    if prob < train_opt['image_distort_strategy']['contrast_prob']:
        contrast_delta = train_opt['image_distort_strategy']['contrast_delta']
        delta = np.random.uniform(-contrast_delta, contrast_delta) + 1
        img = ImageEnhance.Contrast(img).enhance(delta)
    return img


def random_saturation(img):
    prob = np.random.uniform(0, 1)
    if prob < train_opt['image_distort_strategy']['saturation_prob']:
        saturation_delta = train_opt['image_distort_strategy']['saturation_delta']
        delta = np.random.uniform(-saturation_delta, saturation_delta) + 1
        img = ImageEnhance.Color(img).enhance(delta)
    return img


def random_hue(img):
    prob = np.random.uniform(0, 1)
    if prob < train_opt['image_distort_strategy']['hue_prob']:
        hue_delta = train_opt['image_distort_strategy']['hue_delta']
        delta = np.random.uniform(-hue_delta, hue_delta)
        img_hsv = np.array(img.convert('HSV'))
        img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
        img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')
    return img
    

def distort_image(img):
    prob = np.random.uniform(0, 1)
    # Apply different distort order
    if prob > 0.5:
        img = random_brightness(img)
        img = random_contrast(img)
        img = random_saturation(img)
        img = random_hue(img)
    else:
        img = random_brightness(img)
        img = random_saturation(img)
        img = random_hue(img)
        img = random_contrast(img)
    return img


def rotate_image(img):
    """
    图像增强，增加随机旋转角度
    """
    prob = np.random.uniform(0, 1)
    if prob > 0.:
        angle = np.random.randint(-8, 8)
        img = img.convert('RGBA')
        img = img.rotate(angle, resample=Image.BILINEAR, expand=0)
        fff = Image.new('RGBA', img.size, (127, 127, 127, 127))
        img = Image.composite(img, fff, mask=img).convert('RGB')
    return img


def random_expand(img, keep_ratio=True):
    if np.random.uniform(0, 1) < train_opt['image_distort_strategy']['expand_prob']:
        return img

    max_ratio = train_opt['image_distort_strategy']['expand_max_ratio']
    w, h = img.size
    c = 3
    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)
    oh = int(h * ratio_y)
    ow = int(w * ratio_x)
    off_x = random.randint(0, ow - w)
    off_y = random.randint(0, oh - h)

    out_img = np.zeros((oh, ow, c), np.uint8)
    for i in range(c):
        out_img[:, :, i] = train_opt['mean_color']

    out_img[off_y: off_y + h, off_x: off_x + w, :] = img

    return Image.fromarray(out_img)


def preprocess(img, input_size=None):
    img = Image.fromarray(img)
    img_width, img_height = img.size
    img = distort_image(img)
    #img_m = np.mean(img.convert('L'))
    #img_std = max(np.std(img.convert('L')), 1e-2)
   
    img = random_expand(img)
    img = rotate_image(img)
    #img = resize_img(img, input_size)
    #img = img.convert('L')
    #img = (np.array(img).astype('float32') - img_m) / img_std
    img = np.array(img)
    return img