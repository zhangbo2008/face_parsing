#!/usr/bin/python
# -*- encoding: utf-8 -*-
# 这个文件用来做预处理, 我们只是学习. 所以只跑一小部分.
#数据我传到了.https://www.kaggle.com/datasets/ffffffffffffff/celebamask-hq-zip-3-15-gb 可以浏览器下载.
import os.path as osp
import os
import cv2
from transform import *
from PIL import Image

#======出入数据的路径
face_data = 'D:/CelebAMask-HQ/CelebA-HQ-img'
face_sep_mask = 'D:/CelebAMask-HQ/CelebAMask-HQ-mask-anno'


# 写入的路径.
mask_path = 'mask'
counter = 0
total = 0

def main():
    global total
    global counter
    for i in range(15): #======一共15个文件夹

        atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat'] #=一共18个类别.

        for j in range(i * 2000, (i + 1) * 2000):  # 每一个文件夹2000个图片,所以j表示当前图片的索引.
            
            mask = np.zeros((512, 512))

            for l, att in enumerate(atts, 1): #=====对每一个图片有18个特征.
                total += 1
                file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
                path = osp.join(face_sep_mask, str(i), file_name) # 拼出maks的路径.

                if os.path.exists(path):
                    counter += 1
                    sep_mask = np.array(Image.open(path).convert('P')) #=模式“P”为8位彩色图像，它的每个像素用8个bit表示，其对应的彩色值是按照调色板查询出来的。 这样占空空间小一点.
                    # print(np.unique(sep_mask))

                    mask[sep_mask == 225] = l #注意看这里是l不是1, 是L的小写. 是att的索引. 所以我们的mask最后得到的效果是. 图像中skin都是亮度1,.........hat亮度18.  enumerate(atts, 1) 注意索引从0开始. 然后background就是默认的0了!!!!!!
            cv2.imwrite('{}/{}.png'.format(mask_path, j), mask)
            print(j)
            if counter>100:  #===========因为我们只做测试,所以只测100个图片够了.其实只有8个图片. 100个特征小图.
                return 
main()
print(counter, total)