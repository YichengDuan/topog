import habitat_sim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def sample_navigable_points(pathfinder, resolution=0.1):
    x_min, y_min, z_min = pathfinder.get_bounds()[0]
    x_max, y_max, z_max = pathfinder.get_bounds()[1]

    points = []
    x_vals = np.arange(x_min, x_max, resolution)
    z_vals = np.arange(z_min, z_max, resolution)

    for x in x_vals:
        for z in z_vals:
            pos = np.array([x, y_min + 0.1, z])
            if pathfinder.is_navigable(pos):
                points.append(pos)
    return points


def get_untopo_graph(sim:habitat_sim.Simulator, output_path:str, yflip:bool=False):
    
    pathfinder = sim.pathfinder

    # === 1. top-down view map ===
    ground_y = pathfinder.get_bounds()[0][1]
    nav_map = pathfinder.get_topdown_view(
        meters_per_pixel=0.05,
        height=ground_y + 0.1,
        eps=0.5
    )

    if yflip:
        nav_map = np.flipud(nav_map)


    map_h, map_w = nav_map.shape
    print(f"🗺️ Map shape: {map_w}x{map_h}")

    # === 2. 获取所有 navigable points ===
    nav_points = sample_navigable_points(pathfinder, resolution=0.1)
    nav_points = np.array(nav_points)

    # === 3. 将世界坐标映射到地图像素坐标 ===
    meters_per_pixel = 0.05
    bounds_min = pathfinder.get_bounds()[0]  # Vector3
    x_min = bounds_min[0]
    z_min = bounds_min[2]
    grid_origin = (x_min, z_min)
        
    def world_to_map(p):
        x, _, z = p
        mx = int((x - grid_origin[0]) / meters_per_pixel)
        mz = int((z - grid_origin[1]) / meters_per_pixel)
        if yflip:
            mz = map_h - mz
        return mx, mz  

    img = Image.fromarray(np.uint8(255 * nav_map)).convert("RGB")
    draw = ImageDraw.Draw(img)

    for p in nav_points:
        mx, my = world_to_map(p)
        if 0 <= mx < map_w and 0 <= my < map_h:
            draw.point((mx, my), fill=(255, 0, 0)) 

    img.save(output_path)
    print(f"✅ Navpoint map saved to: {output_path}")

    sim.close()

def init_simulator(scene_path):
    # Initialize Habitat-Sim
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = []
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    
    return sim