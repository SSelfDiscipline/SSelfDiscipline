# 创建传感器进行数据的拍摄


import random
import carla
import queue
import threading
import os
# 传感器父类，不会单独创建对象
class Sensor:
    # 客户端，吸附的车辆，保存的路径，传感器相对车辆的位置
    def __init__(self,client,vehicle,re_transform,num,dir):
        self.client = client
        self.vehicle = vehicle
        # 控制图片的标号
        self.frame_id = 1
        self.save_dir = './data/'+dir
        self.re_transform = re_transform
        # 拍摄的总数量
        self.num = num
        self.world = client.get_world()
        # 建议队列,用于数据的规则存取
        self.queue = queue.Queue()
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.finished = False

        # call_back函数
    def check(self):
        if self.frame_id > self.num:
            self.finished = True


# 构建特定属性的RGB相机
class RGBCamara(Sensor):
    RGB_id = 0
    def __init__(self, client, vehicle,re_transform,sensor_list,num,dir):
        super().__init__(client, vehicle,re_transform,num,dir)
        self.save_dir = self.save_dir + '/rgb{}/'.format(RGBCamara.RGB_id)
        RGBCamara.RGB_id += 1
        # 获得指定的蓝图并设置属性
        self.rgb_bp = self.set_attributes()
        self.rgb = self.world.spawn_actor(self.rgb_bp, self.re_transform, attach_to=self.vehicle)
        sensor_list.append(self.rgb)
        # 开启监听模式(我觉得应该属于主进程)
        self.rgb.listen(self.queue.put)
        
    # 得到指定属性（无法通过输入改变参数）的相机
    def set_attributes(self):
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1080')
        camera_bp.set_attribute('image_size_y', '1080')
        # 水平视角为72
        camera_bp.set_attribute('fov', '90')

        return camera_bp
    # 保存图像
    def save(self):
        while not self.queue.empty():
            data = self.queue.get()
            file_path = self.save_dir + '{}.jpg'.format(self.frame_id)
            x = threading.Thread(target=data.save_to_disk, args=(file_path,carla.ColorConverter.Raw))
            x.start()

            self.frame_id += 1
            self.check()


# 语义分割相机,保存两种图片，一种是用于处理，另一种用于呈现可视化
class SEGCamara(Sensor):
    SEG_id = 0
    def __init__(self, client, vehicle,re_transform,sensor_list,num,dir):
        super().__init__(client, vehicle,re_transform,num,dir)
        self.save_dir = self.save_dir + '/seg{}/'.format(SEGCamara.SEG_id)
        SEGCamara.SEG_id += 1
        # 获得指定的蓝图并设置属性
        self.seg_bp = self.set_attributes()
        self.seg = self.world.spawn_actor(self.seg_bp, self.re_transform, attach_to=self.vehicle)
        sensor_list.append(self.seg)
        # 开启监听模式(我觉得应该属于主进程)
        self.seg.listen(self.queue.put)




        # 得到指定属性（无法通过输入改变参数）的相机
    def set_attributes(self):
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', '1080')
        camera_bp.set_attribute('image_size_y', '1080')
        # 水平视角为72
        camera_bp.set_attribute('fov', '90')


        return camera_bp

    # 保存图像
    def save(self):
        while not self.queue.empty():
            data = self.queue.get()
            file_path = self.save_dir + '{}.jpg'.format(self.frame_id)
            VisualSEG(data,file_path)

            self.frame_id += 1
            self.check()


def VisualSEG(data,file_path):
    # 确定路径
    vis_path = file_path.replace('seg','visseg')
    # 保存原始数据
    data.save_to_disk(file_path,carla.ColorConverter.Raw)
    # 保存可视化语义分割
    data.save_to_disk(vis_path,carla.ColorConverter.CityScapesPalette)