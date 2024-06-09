import carla


client = carla.Client('localhost', 2000)
client.set_timeout(2.0)

world = client.get_world()

blueprint_library = world.get_blueprint_library()

blueprint_library = world.get_blueprint_library().find('vehicle.dodge.charger_police_2020')
spawn_points = world.get_map().get_spawn_points()
hero = world.spawn_actor(blueprint_library, spawn_points[0])