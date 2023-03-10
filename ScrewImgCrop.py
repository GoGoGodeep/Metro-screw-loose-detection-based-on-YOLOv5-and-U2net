from PIL import Image
import os
from tqdm import tqdm

# 不检测图像大小
Image.MAX_IMAGE_PIXELS = None

images_dir = r'E:\ScrewData'
outputs_dir = r'E:\outputs'

# 原图片太大，需要对原图片进行裁剪
def crop(images_dir, outputs_dir):

    for img_name in tqdm(os.listdir(images_dir), 'img'):
        im = Image.open(os.path.join(images_dir, img_name))
        _, h = im.size  # (65535, 1808)

        for i in range(22):
            output_path = os.path.join(outputs_dir, img_name[:-4] + '_' + str(i) + '.bmp')
            img_ = im.crop((3000*i, 0, 3000+3000*i, h))
            img_.save(output_path)

        # print("切割成功！")

if __name__ == '__main__':
    crop(images_dir, outputs_dir)