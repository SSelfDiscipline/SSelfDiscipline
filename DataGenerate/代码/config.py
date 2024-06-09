# 配置文件

# 类别字典，类别是根据carla教程中语义分割的类别定义的，value为语义分割图片中类别通道的值，limit为限制识别的物体大小
label_dict = [
    # {'class':'Sidewalk','value':2,'label_id':0,'limit':30},
    {'class':'Building','value':3,'label_id':1,'limit':30},
    {'class':'TrafficLight','value':7,'label_id':2,'limit':30},
    {'class':'TrafficSign','value':8,'label_id':3,'limit':30},
    {'class':'Pedestrian','value':12,'label_id':4,'limit':30},
    {'class':'Rider','value':13,'label_id':5,'limit':30},
    {'class':'Car','value':14,'label_id':6,'limit':30},
    {'class':'Truck','value':15,'label_id':7,'limit':30},
    {'class':'Bus','value':16,'label_id':8,'limit':30},
    {'class':'RoadLine','value':24,'label_id':9,'limit':30},
]