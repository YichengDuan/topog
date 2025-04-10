import habitat_sim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import magnum as mn

def sample_navigable_points(pathfinder, resolution=0.1, y_added=0.0):
    
    x_min, y_min, z_min = pathfinder.get_bounds()[0]
    x_max, _, z_max = pathfinder.get_bounds()[1]

    points = []
    x_vals = np.arange(x_min, x_max, resolution)
    z_vals = np.arange(z_min, z_max, resolution)

    for x in x_vals:
        for z in z_vals:
            query_point = np.array([x, y_min+y_added, z])  # arbitrary high Y           
            snapped = pathfinder.snap_point(point=query_point)
            if snapped:
                points.append(snapped)
    return points

def get_untopo_graph(sim:habitat_sim.Simulator, 
                     output_path:str, 
                     resolution:float=0.25,
                     yflip:bool=False, 
                     semantic_overlay:bool=False):
    
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
    nav_points = sample_navigable_points(pathfinder, resolution=resolution, y_added=0.0)
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
    
    region_colors = {}

    if semantic_overlay:
        semantic_scene = sim.semantic_scene
        for region in semantic_scene.regions:
            name = region.category.name() or "unknown"
            color = region_colors.setdefault(
                name,
                tuple(np.random.randint(50, 200, size=3).tolist())
            )

    # === 4. Draw navpoints with optional region colors ===
    for p in nav_points:
        mx, my = world_to_map(p)
        if 0 <= mx < map_w and 0 <= my < map_h:
            if semantic_overlay:
                label = "unknown"
                for region in sim.semantic_scene.regions:
                    if region.aabb.contains(p):
                        label = region.category.name() or "unknown"
                        break
                color = region_colors.get(label, (255, 0, 0))
                draw.point((mx, my), fill=color)
            else:
                draw.point((mx, my), fill=(255, 0, 0))

    img.save(output_path)
    print(f"✅ Navpoint map saved to: {output_path}")


def draw_semantic_overlay(
    img: Image.Image,
    sim,
    meters_per_pixel: float,
    image_height: int,
    x_min: float,
    z_min: float,
    alpha: int = 60
) -> Image.Image:
    """
    Overlays semantic regions (rooms) on a top-down map as transparent colored rectangles.

    Args:
        img: Base PIL.Image object (top-down grayscale or RGB map).
        sim: habitat_sim.Simulator instance with a loaded semantic scene.
        meters_per_pixel: Map resolution (meters per pixel in top-down view).
        image_height: Height of the top-down image in pixels.
        x_min, z_min: Minimum world coordinates (origin) of the map.
        alpha: Transparency level of region overlays (0-255).

    Returns:
        A new RGBA PIL.Image with semantic regions drawn as transparent overlays.
    """
    semantic_scene = sim.semantic_scene

    # Create transparent overlay
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    region_colors = {}  # Store consistent color for each room type

    def to_map(x, z):
        """Convert world coordinates (x, z) to image pixel coordinates (mx, my)."""
        mx = int((x - x_min) / meters_per_pixel)
        my = image_height - int((z - z_min) / meters_per_pixel)
        return mx, my

    for region in semantic_scene.regions:
        name = region.category.name or "unknown"

        # Assign a unique color per room type
        color = region_colors.setdefault(
            name,
            tuple(np.random.randint(50, 200, size=3).tolist()) + (alpha,)
        )

        # Get AABB corners (project to 2D)
        aabb = region.aabb
        x0, z0 = aabb.min[0], aabb.min[2]
        x1, z1 = aabb.max[0], aabb.max[2]
        mx0, my0 = to_map(x0, z0)
        mx1, my1 = to_map(x1, z1)

        # Draw rectangle overlay
        draw_overlay.rectangle([mx0, my0, mx1, my1], fill=color)

    # Composite overlay on top of the original image
    return Image.alpha_composite(img.convert("RGBA"), overlay)