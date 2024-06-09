import cv2
import os
import numpy as np
import json
import re
from config import *
import math
import random
import xml.etree.ElementTree as ET
data_dir = './data'
def FindPixel(m,value_list):
    maritx_list = []
    for i in range(len(value_list)):
        mask = m == value_list[i]
        maritx_list.append(mask)
    Mask = np.logical_or.reduce(maritx_list)
    return Mask

def FindPixel2(m,value_list):
    maritx_list = []
    for i in range(len(value_list)):
        mask = np.all(m == value_list[i], axis=-1)
        maritx_list.append(mask)
    Mask = np.logical_or.reduce(maritx_list)
    return Mask

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
def merge_boxes(box1, box2):
    """
    将两个矩形合并成一个包含它们的最小矩形
    :param box1: (x1, y1, w1, h1)
    :param box2: (x2, y2, w2, h2)
    :return: 合并后的矩形 (x3, y3, w3, h3)
    """
    loc1 = Convert(box1)
    loc2 = Convert(box2)
    # 计算合并后的矩形的坐标
    x1 = min(loc1[0], loc2[0])
    y1 = min(loc1[1], loc2[1])
    x2 = max(loc1[2], loc2[2])
    y2 = max(loc1[3], loc2[3])
    return (x1, y1, x2-x1, y2-y1)







# 获得图片中目标检测的锚框坐标
def Getbox(seg_path):
    # # 读取图片，特征为BGR
    image = cv2.imread(seg_path)
    # # c,x,y,w,h
    label_box = []
    for label in label_dict:
        # masks = np.all(image[:,:,:-1]  label['value'], axis=-1)
        masks = FindPixel(image[:,:,-1],label['value']) 
        # 查找边界
        contours, _ = cv2.findContours(masks.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > label['limit']]
        bounding_rects = []
        for contour in contours:
            # 左上角的坐标和宽高
            x, y, w, h = cv2.boundingRect(contour)
            bounding_rects.append([label['class'],x, y, w, h])
            # 进行IOU判断和合并
        bounding_rects = merge_boxes_with_high_iou(bounding_rects,label['iou'])
        for rect in bounding_rects:
            label_box.append(rect)
    visseg_path = seg_path.replace('seg', 'visseg')
    image = cv2.imread(visseg_path)
    for label in label_dict2:
        masks = FindPixel2(image,label['value'])
        contours, _ = cv2.findContours(masks.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > label['limit']]
        bounding_rects = []
        for contour in contours:
            # 左上角的坐标和宽高
            x, y, w, h = cv2.boundingRect(contour)
            bounding_rects.append([label['class'],x, y, w, h])
            # 进行IOU判断和合并
        bounding_rects = merge_boxes_with_high_iou(bounding_rects,label['iou'])
        for rect in bounding_rects:
            label_box.append(rect)
    # 返回锚框列表
    return label_box


# 使用递归的思路，将所有IOU值大于阈值的进行合并
def merge_boxes_with_high_iou(boxes, threshold=1.0):
    """
    合并IoU值大于阈值的矩形
    :param boxes: 矩形列表 [(c,x1, y1, x2, y2), ...]
    :param threshold: IoU阈值
    :return: 合并后的矩形列表
    """
    # 如果本身矩形的框数量小于等于1则直接返回
    if len(boxes) <= 1:
        return boxes
    # 如果数量大于1则进行iou判断
    for i in range(len(boxes)-1):
        box1 = boxes[i]
        for j in range(i+1, len(boxes)):
            box2 = boxes[j]
            Iou = ComputerIou(box1[1:],box2[1:])
            # 如果iou值大于阈值
            if Iou >= threshold:
                # 进行合并
                merge_box = merge_boxes(boxes[i][1:],boxes[j][1:])
                # 添加新合并的矩形
                boxes.append([box1[0],merge_box[0],merge_box[1],merge_box[2],merge_box[3]])
                # 这里删除的时候不使用索引，是因为每删除一个标号会改变
                boxes.remove(box1)
                boxes.remove(box2)
                # 进行递归，判断一轮新的矩阵列表是否可以合并
                merge_boxes_with_high_iou(boxes, threshold)

                return boxes
    return boxes







def ComputerIou(box1,box2):
    loc1 = Convert(box1)
    loc2 = Convert(box2)
    # 计算两个框的交集
    inter_x1 = max(loc1[0], loc2[0])
    inter_x2 = min(loc1[2], loc2[2])
    inter_weight = inter_x2 - inter_x1
    inter_y1 = max(loc1[1], loc2[1])
    inter_y2 = min(loc1[3], loc2[3])
    inter_height = inter_y2 - inter_y1
    if inter_height <= 0 or inter_weight <= 0:
        return 0
    # 计算两个框的交集,如果没有相交则为负数
    inter_area = inter_height * inter_weight
    # 判断交集的面积与矩形框的面积的大小关系：
    # 如果交集的面积等于矩形框的面积，即一个矩形包含在另一个矩形内，iou等于1
    if inter_area == box1[2]*box1[3] or inter_area == box2[2]*box2[3]:
        return 1
    # 如果交集的面积为0或者负数，说明没有相交，iou等于0
    if inter_area <= 0:
        return 0
    # 相交
    else:
        # 计算两个框的并集
        union_area = box1[2]*box1[3] + box2[2]*box2[3] - inter_area
        # 返回Iou
        return inter_area/union_area





# 对中心宽高坐标改为左上角和右下角坐标
def Convert(box):
    x,y,w,h = box
    # 计算左上坐标和右下坐标
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return x1,y1,x2,y2

# 绘制边框
def DrawCounts(rgb_path,seg_path):
    image = cv2.imread(rgb_path)
    label_box = Getbox(seg_path)
    for box in label_box:
        x,y,w,h = box[1:]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, box[0], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('image', image)
    cv2.waitKey(0)
    # 保存图片
    # image = cv2.imread(rgb_path)
    # cv2.imwrite('./question/1.jpg',image)


# 观察每一个绘制后的图像
def Watch(data_dir):

    rgb_list, seg_list,visseg_list = load_image_dir(data_dir)
    index = [x for x in range(len(rgb_list))]
    # 打乱顺序
    random.shuffle(index)
    # rgb_list, seg_list,visseg_list = rgb_list[index], seg_list[index],visseg_list[index]
    rgb_list = [rgb_list[i] for i in index]
    seg_list = [seg_list[i] for i in index]
    visseg_list = [visseg_list[i] for i in index]
    for i in range(len(rgb_list)):
        DrawCounts(rgb_list[i],seg_list[i])


# 加载像素
# def hehe(masks):
#     a = masks.astype(np.uint8)
#     index = np.where(a == 1)
#     a[index] = 255
#     cv2.imshow('a',a)
#     cv2.waitKey(0)

# 转化xml格式
# label_path为存储xml的路径,label_box为Getbox函数
def RecordYoloXml(label_path,label_box):
    with open(label_path, 'w') as f:
        f.write('<annotation>\n')
        f.write("   <path>{}</path>\n".format(label_path.replace('label','data').replace('xml','jpg')))
        f.write("   <bbox_num>{}</bbox_num>\n".format(len(label_box)))
        f.write("   <size>\n")
        f.write("       <width>{}</width>\n".format(1080))
        f.write("       <height>{}</height>\n".format(1080))
        f.write("       <depth>{}</depth>\n".format(3))
        f.write("   </size>\n")

        for box in label_box:
            f.write("   <object>\n")
            f.write("       <name>{}</name>\n".format(box[0]))
            f.write("       <label_id>{}</label_id>\n".format(name_id[box[0]]))
            f.write("       <bndbox>\n")
            f.write("           <xmin>{}</xmin>\n".format(box[1]))
            f.write("           <ymin>{}</ymin>\n".format(box[2]))
            f.write("           <xmax>{}</xmax>\n".format(box[3]+box[1]))
            f.write("           <ymax>{}</ymax>\n".format(box[4]+box[2]))
            f.write("       </bndbox>\n")
            f.write("   </object>\n")

        f.write('</annotation>\n')












# 进行数据的处理，包括：1.删除无目标的图像 2.将有目标的图像的目标进行记录 3.将数据划分为训练集和测试集
def Process(rgb_list,train_weight = 0.8):
    index = [x for x in range(len(rgb_list))]
    print('正在写入xml')
    for i in range(len(rgb_list)):
        label_box = Getbox(rgb_list[i].replace('rgb0','seg0'))
        # 删除无目标的图像
        if len(label_box) > 0:
            label_path = rgb_list[i].replace('data','label').replace('jpg','xml')
            b = label_path.split('\\')
            labels_path = os.path.join(b[0], b[1], b[2])
            # 如果不存在则创建
            if not os.path.exists(labels_path):
                os.makedirs(labels_path)
            # 转化xml格式
            RecordYoloXml(label_path,label_box)
        else:
            index.remove(i)
    print('写入xml完成')
    # 进行删除
    rgb_list = [rgb_list[i] for i in index]
    print('删除操作完成,剩下的图像数量为：',len(rgb_list))
    # 划分训练集和测试集
    random.shuffle(rgb_list)
    train_list = rgb_list[:int(len(rgb_list)*train_weight)]
    test_list = rgb_list[int(len(rgb_list)*train_weight):]
    if not os.path.exists('./split'):
        os.makedirs('./split')
    # 写入训练集和测试集
    with open('./split/train.txt','w') as f:
        for train in train_list:
            train = os.path.abspath(train)
            f.write(train+'\n')
    with open('./split/test.txt','w') as f:
        for test in test_list:
            test = os.path.abspath(test)
            f.write(test+'\n')
    print('划分完成')

    print("数据处理完成")


def xywh_to_xywh(size,box):
    """将边界框坐标从VOC格式转换为YOLO格式"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    w = box[2] - box[0]
    h = box[3] - box[1]
    # 保留6位小数
    x = round(x * dw,6)
    w = round(w * dw,6)
    y = round(y * dh,6)
    h = round(h * dh,6)
    return (x, y, w, h)




# 将VOC格式的标签转化为YOLO格式的标签
def ConvertYolo(split_path = './split',save_path = './yololabel'):
    train_path = split_path+'/train.txt'
    test_path = split_path+'/test.txt'
    # 如果没有则创建文件
    save_train = save_path+'/train'
    save_test = save_path+'/test'
    if not os.path.exists(save_train):
        os.makedirs(save_train)
    if not os.path.exists(save_test):
        os.makedirs(save_test)
    # 读取图片
    with open(train_path,'r') as f:
        train_list = f.readlines()
        for i in range(len(train_list)):
            # xml的路径,去掉换行符
            xml_path = train_list[i].replace('data','label').replace('jpg','xml').strip('\n')
            # 保存的txt路径
            txt_path = save_train+'/'+str(i)+'.txt'
            # 读取xml
            tree = ET.parse(xml_path)
            root = tree.getroot()
            # 获取图片尺寸
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            # 获取目标
            # yolo_boxs = []
            for obj in root.iter('object'):
                bbox = obj.find('bndbox')
                b = (
                    int(bbox.find('xmin').text),
                    int(bbox.find('ymin').text),
                    int(bbox.find('xmax').text),
                    int(bbox.find('ymax').text),
                )
                label_id = obj.find('label_id').text

                bb = xywh_to_xywh((w, h), b)
                # yolo_boxs.append((label_id, bb))
                # 不进行覆盖。继续在后面写
                with open(txt_path, 'a') as f:
                    # for yolo_box in yolo_boxs:
                    #     label_id, bb = yolo_box
                    f.write(f'{label_id} {bb[0]} {bb[1]} {bb[2]} {bb[3]}\n')


    # 对于训练集也是这样
    with open(test_path,'r') as f:
        test_list = f.readlines()
        for i in range(len(test_list)):
            # xml的路径
            xml_path = test_list[i].replace('data','label').replace('jpg','xml').strip('\n')
            # 保存的txt路径
            txt_path = save_test+'/'+str(i)+'.txt'
            # 读取xml
            tree = ET.parse(xml_path)
            root = tree.getroot()
            # 获取图片尺寸
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            # 获取目标
            # yolo_boxs = []
            for obj in root.iter('object'):
                bbox = obj.find('bndbox')
                b = (
                    int(bbox.find('xmin').text),
                    int(bbox.find('ymin').text),
                    int(bbox.find('xmax').text),
                    int(bbox.find('ymax').text),
                )
                label_id = obj.find('label_id').text

                bb = xywh_to_xywh((w, h), b)
                # yolo_boxs.append((label_id, bb))
                # 不进行覆盖。继续在后面写
                with open(txt_path, 'a') as f:
                    # for yolo_box in yolo_boxs:
                    #     label_id, bb = yolo_box
                    f.write(f'{label_id} {bb[0]} {bb[1]} {bb[2]} {bb[3]}\n')




# 将txt中的相对路径转为绝对路径
def Abspath(txt_path):
    abs_list = []
    # 读取train.txt
    with open(txt_path,'r') as f:
        train_list = f.readlines()

        # 将相对路径改为绝对路径
        for i in range(len(train_list)):
            # 获取图片的相对路径
            img_path = train_list[i].strip('\n')
            # 获取绝对路径
            img_path = os.path.abspath(img_path)
            # 获取图片的标签
            abs_list.append(img_path)
    with open(txt_path,'w') as f:
        for i in range(len(abs_list)):
            f.write(f'{abs_list[i]}\n')














if __name__ == '__main__':
    
    # Watch(data_dir)
    path = './data/Town10/rgb0/842.jpg'   
    seg_path = path.replace('rgb0','seg0')
    # vis = path.replace('rgb0','visseg0')
    # image = cv2.imread(path)
    # cv2.imshow('image',image)
    # # cv2.waitKey(0)
    DrawCounts(path,seg_path)
    # rgb_list, seg_list,visseg_list = load_image_dir(data_dir)
    # # 删除无目标的图像
    # CheckObjects(rgb_list,seg_list,visseg_list)
    # Process(rgb_list)
    # ConvertYolo()
    # Abspath('./split/train.txt')
    # Abspath('./split/test.txt')
    # pass




