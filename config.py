# 配置文件

# 类别字典，类别是根据carla教程中语义分割的类别定义的，value为语义分割图片中类别通道的值，limit为限制识别的物体大小
label_dict = [
    # {'class':'Sidewalk','value':2,'label_id':0,'limit':30},
    # {'class':'Building','value':[3],'label_id':0,'limit':30},
    # {'class':'TrafficSign','value':[8],'label_id':2,'limit':50},
    # {'class':'Pedestrian','value':12,'label_id':4,'limit':30},
    # {'class':'Motorcycle','value':[18],'label_id':3,'limit':100,'iou':0.01},
    # {'class':'Bicycle','value':[19],'label_id':3,'limit':20,'iou':0.01},
    # {'class':'Rider','value':[18,19,13,12],'label_id':3,'limit':40,'iou':0.01},
    # {'class':'行人','value':[12],'label_id':3,'limit':30},
    # {'class':'Car','value':[14],'label_id':4,'limit':50},
    # {'class':'Truck','value':15,'label_id':7,'limit':30},
    # {'class':'Bus','value':16,'label_id':8,'limit':30},
    # {'class':'RoadLine','value':[24],'label_id':9,'limit':50},
]

# traffic_light类在通过实验得知，原分割图像的标签通道值为0，在处理后的可视化分割图片的色素质为（250，170，31）rgb
label_dict2 = [
    {'class':'TrafficLight','value':[(31,170,250)],'label_id':0,'limit':100,'iou':0.5},
    {'class':'TrafficSign','value':[(0,220,220)],'label_id':1,'limit':100,'iou':0.5},
    # 汽车类
    # 轿车
    {'class':'Vehicle','value':[(142,0,0)],'label_id':2,'limit':150,'iou':1.0},
    # 公交车
    {'class':'Vehicle','value':[(100,60,0)],'label_id':2,'limit':150,'iou':1.0},
    # 货车
    {'class':'Vehicle','value':[(70,0,0)],'label_id':2,'limit':150,'iou':1.0},
    # 骑行者类
    {'class':'Rider','value':[(60,20,220),(0,0,254),(230,0,0),(32,11,119)],'label_id':3,'limit':50,'iou':0.01},
    # 建筑类
    # {'class':'Building','value':[(70,70,70)],'label_id':2,'limit':1500,'iou':1.1},
    # 马路线
    {'class':'Roadline','value':[(50,234,156)],'label_id':4,'limit':200,'iou':1.1},
    # 人行道和非机动车道
    # {'class':'Sidewalk','value':[(232,35,244)],'label_id':2,'limit':100,'iou':1.1},
    # 障碍物
    # {'class':'Fence','value':[(153,153,190)],'label_id':2,'limit':50,'iou':1.1},


]


# # 组合体的IOU
# Combo_IOU = 0.01
# # 单体的IOU
# Single_IOU = 0.5

# 图像的形状:
height = 1080
width = 1920
# 图像通道数
channel = 3


name_id = {
    'TrafficSign':1,
    'Rider':3,
    'Roadline':4,
    'TrafficLight':0,
    'Vehicle':2,
}


town_list = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town10']
