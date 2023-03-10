# 检测全流程

### 第零步：先对地铁图片进行裁剪（ScrewImgCrop.py），并做亮度增强、图像旋转（ScrewStrengthen.py），再打标签并转成yolo的格式（images_tag.py、voc_to_yolo.py），保存到Screwdataset文件夹

### 第一步：使用yolo进行训练，使用yolov5x.pt

### 第二步：使用yolo进行目标侦测（detect.py），并输出labels（runs/detect/exp/labels）

### 第三步：根据labels和原始图像进行螺丝裁剪（ScrewSingleCrop.py），保存到MaskData\images文件夹中

### 第四步：对螺丝图片（MaskData\images）进行直方图均衡化（ScrewStrengthen.py），并做图像分割的标签（MaskData\mask）并二值化（ScrewStrengthen.py），保存到MaskData\binary_mask文件夹中

### 第五步：构建U2net并进行训练

### 第六步：使用训练好的U2net网络对Creaw\images中的图片进行分割并保存结果至Creaw\outputs

### 第七步：使用opencv得到螺丝的点（ScrewAngle.py），获取坐标并计算角度