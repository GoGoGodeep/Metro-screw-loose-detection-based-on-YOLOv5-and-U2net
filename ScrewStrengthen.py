from PIL import Image
import os
from skimage import exposure, img_as_float, io
import numpy as np
from tqdm import tqdm
import cv2

# 不检测图像大小
Image.MAX_IMAGE_PIXELS = None

# 图像旋转
def imgRotate(images_dir):

    for path in tqdm(os.listdir(images_dir), 'path'):
        for img_name in tqdm(os.listdir(os.path.join(images_dir, path)), 'img'):
            im = Image.open(os.path.join(images_dir, path, img_name))

            w, h = im.size
            # 如果图片的宽比高小，说明图片反了
            if w < h:
                im_rotate = im.transpose(Image.Transpose.ROTATE_90)
                im_rotate.save(os.path.join(images_dir, path, img_name[:-4] + '.bmp'))

    print("翻转成功！")

# 亮度增强
def lightUp(images_dir, outputs_dir):

    for path in tqdm(os.listdir(images_dir), 'path'):
        for img_name in tqdm(os.listdir(os.path.join(images_dir, path)), 'img'):
            im = Image.open(os.path.join(images_dir, path, img_name))

            # 转换为 skimage 可操作的格式
            img = img_as_float(im)

            # 调整图像亮度，数值低于1.0，表示调亮；高于1.0表示调暗。
            img_light = exposure.adjust_gamma(img, 0.7)

            # 对图像数据进行缩放和舍入操作，并将其转换为无符号8位整数类型
            # 不做这一步图片会有损失
            img_light = np.round(img_light * 255).astype(np.uint8)

            # 存储文件到新的路径中，并修改文件名
            io.imsave(os.path.join(outputs_dir, img_name[:-4] + '.bmp'), img_light)

    print("增亮成功！")

# 直方图均衡化进行图像增强
def clash(Images_dir):

    for i in tqdm(os.listdir(Images_dir)):
        img = cv2.imread(os.path.join(Images_dir, i), 0)

        # 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        dst = clahe.apply(img)

        cv2.imwrite(os.path.join(Images_dir, i), dst)

# 分割标签二值化
def binary(img_dir, output_dir):

    for json_name in os.listdir(img_dir):
        for label_name in os.listdir(os.path.join(img_dir, json_name)):
            # 打开标签图像文件
            try:
                if 'label' in label_name and '_' not in label_name:
                    img = Image.open(os.path.join(img_dir, json_name, label_name))

                    # 将图像转换为黑白图像
                    img = img.convert('L')

                    # 将黑白图像进行二值化处理
                    threshold = 10
                    img = img.point(lambda x: 255 if x > threshold else 0)

                    # 保存二值化后的图像
                    img.save(os.path.join(output_dir, json_name[:-5] + '.png'))
            except:
                pass


if __name__ == '__main__':
    # images_dir = r'E:\data'
    # outputs_dir = r'E:\ScrewData'

    # imgRotate(images_dir)

    # lightUp(images_dir, outputs_dir)

    clash(r'C:\Users\Administrator\Desktop\Creaw\images')

    # img_dir = r'E:\MaskData\mask'
    # output_dir = r'E:\MaskData\binary_mask'
    # binary(img_dir, output_dir)

