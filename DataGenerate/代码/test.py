import cv2
import os
import numpy as np
import json
import re
from config import *

data_dir = './data'
# 获得指定路径中所有地图的rgb数据
def load_image_dir(data_dir):
    rgb_list = []
    seg_list = []
    visseg_list = []
    town_list = os.listdir(data_dir)
    for town in town_list:
        datas_list = os.listdir(os.path.join(data_dir, town))
        # 获取存放数据的文件夹
        datas_list = [x for x in datas_list if x.startswith('rgb') and os.path.isdir(os.path.join(data_dir, town, x))]
        # 获取图片的路径
        for datas in datas_list:
            datas_dir = os.path.join(data_dir, town, datas)
            data_list = os.listdir(datas_dir)
            for data in data_list:
                if data.endswith('.jpg'):
                    rgb_list.append(os.path.join(datas_dir, data))
                    seg_list.append(os.path.join(datas_dir, data).replace('rgb', 'seg'))
                    visseg_list.append(os.path.join(datas_dir, data).replace('rgb', 'visseg'))

    return rgb_list, seg_list,visseg_list

# 获得图片中目标检测的锚框坐标
def Getbox(seg_path):
    # 读取图片，特征为BGR
    image = cv2.imread(seg_path)
    # c,x,y,w,h
    label_box = []
    for label in label_dict:
        masks = image[:,:,-1]==label['value']
        # 查找边界
        contours, _ = cv2.findContours(masks.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > label['limit']]

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            label_box.append((label['class'],x, y, w, h))
    # 返回锚框列表
    return label_box


# 绘制边框
def DrawCounts(image_path,label_box):
    image = cv2.imread(image_path)
    for box in label_box:
        x,y,w,h = box[1:]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, box[0], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('image', image)
    cv2.waitKey(0)


def Watch(data_dir):
    rgb_list, seg_list,visseg_list = load_image_dir(data_dir)
    for i in range(len(rgb_list)):
        label_box = Getbox(seg_list[i])
        DrawCounts(rgb_list[i],label_box)

if __name__ == '__main__':
    
    Watch(data_dir)
                





