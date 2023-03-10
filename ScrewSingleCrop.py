import os
from PIL import Image

def SingleCrop(txt_dir, img_dir, img_name):
    with open(txt_dir) as f:
        im = Image.open(img_dir)#.convert('RGB')

        num = 0
        while True:
            idxs = f.readline()
            if idxs == '':
                break
            idxs = idxs.strip().split(' ')
            # print(idxs)

            x1 = float(idxs[1]) - float(idxs[3]) / 2
            y1 = float(idxs[2]) - float(idxs[4]) / 2
            x2 = float(idxs[1]) + float(idxs[3]) / 2
            y2 = float(idxs[2]) + float(idxs[4]) / 2

            x1 = x1 * im.size[0]
            x2 = x2 * im.size[0]
            y1 = y1 * im.size[1]
            y2 = y2 * im.size[1]

            img = im.crop((x1, y1, x2, y2))
            img.save(os.path.join(r'C:\Users\Administrator\Desktop\Creaw\images', img_name + '_{0}'.format(num) + '.bmp'))

            num += 1

        print("螺丝图片切割成功！")

if __name__ == '__main__':

    exp_dir = 'runs/detect/exp2'
    img_dir = 'data/images'    # 用原图进行抠图，否则会有红边

    try:
        for img_name in os.listdir(exp_dir):
            img = os.path.join(img_dir, img_name[:-4] + '.bmp')
            labels = os.path.join(exp_dir, 'labels', img_name[:-4] + '.txt')

            SingleCrop(labels, img, img_name[:-4])
    except:
        pass