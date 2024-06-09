# 本科毕业设计

## 代码介绍：

- DataGenerate文件：用来连接Carla生成交通场景和进行数据采集
  1. main.py:主文件
  2. generate_traffic.py:生成交通流
  3. We.py:同步模式周期变化天气
  4. Sensor.py:设置生成特定属性的传感器，包含自定义类
- autolabel.py:进行数据的自动标注，数据标签的保存，yolo和voc格式的转换，以及查看标注效果
- config.py:设置自动标注的类别以及限制
- yolov5master文件：进行对于仿真数据集的网络训练和检测
  1. train.py:进行训练
  2. detect.py:进行预测

## 几个结果：

- ./yolov5master/runs/train/exp68:为本模型的最终训练结果
- ./yolov5master/runs/detect/exp20：为20张图片和一个视频的预测结果

## 数据集的下载：

