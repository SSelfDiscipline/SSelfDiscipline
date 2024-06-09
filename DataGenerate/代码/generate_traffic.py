import carla
import random
# 创建交通管理器
def create_traffic_manager(client,port = 2500):
    traffic_manager = client.get_trafficmanager(port)
    # 车辆之间的最小距离
    traffic_manager.set_global_distance_to_leading_vehicle(3.0)
    # 重生车辆
    traffic_manager.set_respawn_dormant_vehicles(True)
    # 生成距离
    traffic_manager.set_boundaries_respawn_dormant_vehicles(25,50)
    # 混合物理模式
    traffic_manager.set_hybrid_physics_mode(True)
    # 设置混合物理模式的半径
    traffic_manager.set_hybrid_physics_radius(50.0)
    # 设置速度
    traffic_manager.global_percentage_speed_difference(30.0)


    return traffic_manager

















# 连接服务器的客户端,交通管理器
# 车辆，行人，行人管理器的客户端
# 车辆和行人的数量
def spawn_actor(client,traffic_manager,vehicle_list,walker_list,walker_control_list,vehicle_number = 10, walker_number = 10):
    world = client.get_world()
    # 生成车辆
    vehicle_bp_list = world.get_blueprint_library().filter("vehicle.*")
    # 转换为列表
    # 筛选出具有车灯的车辆
    # vehicle_bp_list = [x for x in vehicle_bp_list if 
    #                 #    正常访问无法得到该const常量，但是通过字符串可以提取出来
    #                     str(x.get_attribute('has_lights')).split('=')[-1].split('(')[0] == 'True']
    # 判断可生成点的数量
    spawn_points = world.get_map().get_spawn_points()
    if vehicle_number >len(spawn_points):
        print("生成车辆数量过多，调整为{}辆".format(len(spawn_points)))
        vehicle_number = len(spawn_points)

    for i in range(vehicle_number):
        vehicle_bp = random.choice(vehicle_bp_list)
        if vehicle_bp.has_attribute('color'):
            vehicle_bp.set_attribute('color', random.choice(vehicle_bp.get_attribute('color').recommended_values))
        # 避免碰撞
        spawn_point = random.choice(spawn_points)
        spawn_points.remove(spawn_point)
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle_list.append(vehicle)
  
        





        # 生成行人
    run_p = 0.2
    cross_p = 0.1
    world.set_pedestrians_cross_factor(cross_p)
    world.set_pedestrians_seed(16)
    walker_bp_list = world.get_blueprint_library().filter("walker.*")
    walker_bp_list = [x for x in walker_bp_list]
    spawn_points = []
    for i in range(walker_number):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        loc.z += 1.0
        if loc!=None:
            spawn_point.location = loc
            spawn_points.append(spawn_point)
        
    
    speed_list = []
    for i in range(len(spawn_points)):
        walker_bp = random.choice(walker_bp_list)
        walker = world.spawn_actor(walker_bp,spawn_points[i])
        walker_list.append(walker)
        # 可被碰撞
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        if walker_bp.has_attribute('speed'):
            if random.random()<run_p:
                speed_list.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                speed_list.append(walker_bp.get_attribute('speed').recommended_values[1])
        else:
            speed_list.append(0.0)

    for i in range(walker_number):
        # 获取行人控制器
        controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker_list[i])
        # 设置行人的目标
        controller.start()
        controller.go_to_location(world.get_random_location_from_navigation())
        controller.set_max_speed(float(speed_list[i]))
        walker_control_list.append(controller)








        # 设置为同步模式
        # settings = world.get_settings()
        # traffic_manager.set_synchronous_mode(True)
        # synchronous_master = True
        # settings.synchronous_mode = True
        # settings.fixed_delta_seconds = 0.05
        # world.apply_settings(settings)










    #     while True:
    #         world.tick()
    #         # spectator = world.get_spectator()
    #         # vehicle_transform = vehicle_list[0].get_transform()
    #         # vehicle_transform.location.z += 5
    #         # spectator.set_transform(vehicle_transform)
    # finally:
    #     # 销毁车辆
    #     client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])
    #     client.apply_batch([carla.command.DestroyActor(x) for x in walker_list])
    #     client.apply_batch([carla.command.DestroyActor(x) for x in walker_control_list])
    #     # 退回异步模式
    #     settings = world.get_settings()
    #     settings.synchronous_mode = False
    #     settings.fixed_delta_seconds = None
    #     world.apply_settings(settings)
    

if __name__ == '__main__':
    vehicle_list = []
    walker_list = []
    walker_control_list = []
    try:

        pass
    finally:
        pass