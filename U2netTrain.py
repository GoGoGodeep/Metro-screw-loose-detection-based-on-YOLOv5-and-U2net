import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from PIL import Image
import os
from tqdm import tqdm
from U2net import U2NET
from U2net import U2NETP

# ------- 1. define loss function --------
bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
    loss0.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item()))

    return loss0, loss


# ------- 2. set the directory of training dataset --------
# 使用u2net
model_name = 'u2net'

model_dir = 'U2net_model/'

epoch_num = 2000
batch_size_train = 12
batch_size_val = 1
train_num = 0
val_num = 0

def mask():
    tra_dir = r"E:\MaskData"
    # 对标签图片与原图进行二值化处理
    for i in tqdm(os.listdir('E:\MaskData\mask')):
        tra_image = Image.open(os.path.join(tra_dir, 'mask', i))
        binary = tra_image.convert("1")
        binary.save(os.path.join(tra_dir, 'binary_mask', i))

# mask()

# 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

class Dataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, "binary_mask"))  # 标签

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        name = self.name[index]
        name_bmp = name[:-3] + "bmp"
        img_path = os.path.join(self.path, "images")  # 原始图片
        label_path = os.path.join(self.path, "binary_mask")  # 标签

        img = Image.open(os.path.join(img_path, name_bmp))
        label = Image.open(os.path.join(label_path, name))

        # 等比缩放
        bg_img = torchvision.transforms.ToPILImage()(torch.zeros(3, 224, 224))
        bg_label = torchvision.transforms.ToPILImage()(torch.zeros(1, 224, 224))

        img_size = torch.Tensor(img.size)
        # 获取最大边长的索引
        l_max_index = img_size.argmax()
        ratio = 224 / img_size[l_max_index]
        img_resize = img_size * ratio
        img_resize = img_resize.long()

        img_use = img.resize(img_resize)
        label_use = label.resize(img_resize)

        bg_img.paste(img_use)
        bg_label.paste(label_use)
        return transform(bg_img), transform(bg_label)


data_loader = DataLoader(Dataset(r"E:\MaskData"), batch_size=10, shuffle=True)
# print(data_loader)


# ------- 3. define model --------
# define the net
if (model_name == 'u2net'):
    net = U2NET(3, 1)
elif (model_name == 'u2netp'):
    net = U2NETP(3, 1)

if torch.cuda.is_available():
    net.cuda()

net.load_state_dict(torch.load('U2net_model/u2net.pt'))
print("权重加载成功")

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 500  # save the model every 2000 iterations

for epoch in range(0, epoch_num):
    net.train()

    for i, (inputs, labels) in enumerate(data_loader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        # inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        running_tar_loss += loss2.item()

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
        epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,
        running_tar_loss / ite_num4val))

        if ite_num % save_frq == 0:
            torch.save(net.state_dict(), model_dir + "train_%3f_tar_%3f.pt" % (
            running_loss / ite_num4val, running_tar_loss / ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0
