import os
import torch
import torchvision
from U2net import U2NET
from PIL import Image
import numpy as np


net = U2NET().cuda()
net.load_state_dict(torch.load('U2net_model/u2net.pt'))
net.eval()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

def segment(img_dir, out_dir):

    for img_name in os.listdir(img_dir):
        im = Image.open(os.path.join(img_dir, img_name))
        # 等比缩放
        bg_img = torchvision.transforms.ToPILImage()(torch.zeros(3, 224, 224))

        img_size = torch.Tensor(im.size)
        # 获取最大边长的索引
        l_max_index = img_size.argmax()
        ratio = 224 / img_size[l_max_index]
        img_resize = img_size * ratio
        img_resize = img_resize.long()

        img_use = im.resize(img_resize)

        bg_img.paste(img_use)
        img_data = transform(bg_img).unsqueeze_(0).cuda()

        d0, _, _, _, _, _, _ = net(img_data)

        pred = d0.detach().cpu().numpy()
        pred = np.squeeze(pred)
        img = Image.fromarray((pred * 255).astype(np.uint8))

        img.save(os.path.join(out_dir, img_name))


if __name__ == '__main__':
    img_dir = r'C:\Users\Administrator\Desktop\Creaw\images'
    out_dir = r'C:\Users\Administrator\Desktop\Creaw\outputs'
    segment(img_dir, out_dir)