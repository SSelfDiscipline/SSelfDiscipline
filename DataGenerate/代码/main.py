from We import *
from generate_traffic import *
import carla
import random
from sensor      import *

town_list = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town10']


def main(host='localhost', port=2000,town =town_list[-1]):
    client = carla.Client(host, port)
    # 设置连接等待的最大时间
    client.set_timeout(20.0)
    # 天气进行改变的一个最小时间段
    change_time = 0.1
    # 天气改变的速度
    speed_factor = 1.5
    vehicle_list = []
    walker_list = []
    walker_controller_list = []
    sensor_list = []
    world = client.get_world()
    # client.load_world(town)
    try:
        # 设置为同步模式
        # 生成交通管理器
        traffic_manager = create_traffic_manager(client,2500)
        settings = world.get_settings()
        traffic_manager.set_synchronous_mode(True)
        
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.10
        world.apply_settings(settings)
        blueprint_library = world.get_blueprint_library().find('vehicle.dodge.charger_police_2020')
        blueprint_library.set_attribute('role_name', 'hero')
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        hero = world.spawn_actor(blueprint_library, spawn_points[0])
        vehicle_list.append(hero)

        # 生成车辆和行人actor
        spawn_actor(client,traffic_manager,vehicle_list,walker_list,walker_controller_list,24,5)
        # 设置车辆自动驾驶并且自动更新车灯
        for x in vehicle_list:
            x.set_autopilot(True,traffic_manager.get_port())
            traffic_manager.update_vehicle_lights(x, True)
            traffic_manager.ignore_lights_percentage(x,100)
            traffic_manager.ignore_signs_percentage(x,100)
        # 生成传感器
        # 正面传感器
        cam0_transform = carla.Transform(carla.Location(x=1.8, y=0, z=1.90), carla.Rotation(pitch=0, yaw=0, roll=0))

        # RGB0 = RGBCamara(client,hero,cam0_transform,sensor_list,1600,town)

        # SEG0 = SEGCamara(client,hero,cam0_transform,sensor_list,1600,town)

        # 设置时间的周期性变化
            
        weather = Weather(world.get_weather())
        elapsed_time = 0.0


        # 获取观察者视角
        spector = world.get_spectator()
        # 循环
        flag = True
        print('开始记录数据')
        while flag:

            # 30为超时时间
            world.tick()
            # 快照的时间戳
            # 处理CARLA服务器返回的数据
            timestamp = world.get_snapshot().timestamp
            elapsed_time += timestamp.delta_seconds
            
            # 检查是否需要更新天气状态
            if elapsed_time > change_time:
                weather.tick(speed_factor * elapsed_time)
                world.set_weather(weather.weather)
                sys.stdout.write('\r' + str(weather) + 12 * ' ')
                sys.stdout.flush()
                elapsed_time = 0.0

            tr = hero.get_transform()
            tr.location.z +=1.8
            tr.location.x += 1.0
            # tr.rotation.yaw += 45
            spector.set_transform(tr)
            # RGB0.save()
            # SEG0.save()

            # if RGB0.finished and  SEG0.finished :
            #     flag = False
    finally:
        # 销毁车辆
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])
        client.apply_batch([carla.command.DestroyActor(x) for x in walker_list])
        client.apply_batch([carla.command.DestroyActor(x) for x in walker_controller_list])
        client.apply_batch([carla.command.DestroyActor(x) for x in sensor_list])
        # 退回异步模式
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)



if __name__ == '__main__':
    # for town in town_list:
    #     main(town=town)
    # main(town = town_list[3])
    main()