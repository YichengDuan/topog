#!/usr/bin/env python3
"""
A non-interactive Habitat-Sim viewer that displays the agent's view using Magnum.
It supports simple movement and camera rotation commands (with no user input)
and can save the current view to an image file.
"""

import os
import time
import numpy as np
import magnum as mn
from magnum import shaders, text
import queue
from magnum.platform.glfw import Application
import threading
import habitat_sim
from habitat_sim import physics
from habitat_sim.utils.settings import default_sim_settings, make_cfg
import habitat_sim.agent
# Create a global thread-safe queue for commands.
command_queue = queue.Queue()



class HabitatSimNonInteractiveViewer(Application):
    def __init__(self, sim_settings):
        # Set up the GLFW window configuration.
        window_size = mn.Vector2(sim_settings["window_width"], sim_settings["window_height"])
        configuration = self.Configuration()
        configuration.title = "Habitat-Sim Non-Interactive Viewer"
        configuration.size = window_size
        super().__init__(configuration)

        self.fps = 60.0
        # Calculate sensor resolution based on the window size.
        camera_resolution = mn.Vector2(self.framebuffer_size)
        sim_settings["width"] = camera_resolution.x
        sim_settings["height"] = camera_resolution.y

        self.sim_settings = sim_settings
        # Create the Habitat-Sim configuration and override the default agent.
        self.cfg = make_cfg(sim_settings)
        self.agent_id = sim_settings.get("default_agent", 0)
        self.cfg.agents[self.agent_id] = self.default_agent_config()
        self.sim = habitat_sim.Simulator(self.cfg)
        self.agent = self.sim.get_agent(self.agent_id)
        # Assume the agent has a sensor named "color_sensor".
        self.render_camera = self.agent.scene_node.node_sensor_suite.get("color_sensor")

        # Set up Magnum text to display a simple title.
        self.display_font = text.FontManager().load_and_instantiate("TrueTypeFont")
        relative_path_to_font = "../data/fonts/ProggyClean.ttf"
        font_path = os.path.join(os.path.dirname(__file__), relative_path_to_font)
        self.display_font.open_file(font_path, 13)
        self.glyph_cache = text.GlyphCacheGL(mn.PixelFormat.R8_UNORM, mn.Vector2i(256), mn.Vector2i(1))
        self.display_font.fill_glyph_cache(
            self.glyph_cache,
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        )
        self.window_text = text.Renderer2D(self.display_font, self.glyph_cache, 16.0, text.Alignment.TOP_LEFT)
        self.window_text.reserve(256)
        self.window_text_transform = (
            mn.Matrix3.projection(self.framebuffer_size)
            @ mn.Matrix3.translation(mn.Vector2(self.framebuffer_size) * mn.Vector2(-0.49, 0.49))
        )
        self.shader = shaders.VectorGL2D()

    def default_agent_config(self):
        """
        Create a default agent configuration with movement and camera rotation actions.
        """
        make_action_spec = habitat_sim.agent.ActionSpec
        make_actuation_spec = habitat_sim.agent.ActuationSpec
        MOVE, LOOK = 0.07, 1.5
        actions = [
            "move_left", "turn_left", "move_right", "turn_right",
            "move_backward", "look_up", "move_forward", "look_down",
            "move_down", "move_up",
        ]
        action_space = {}
        for action in actions:
            actuation_value = MOVE if "move" in action else LOOK
            action_space[action] = make_action_spec(action, make_actuation_spec(actuation_value))
        sensor_spec = self.cfg.agents[self.agent_id].sensor_specifications
        return habitat_sim.agent.AgentConfiguration(
            height=1.5,
            radius=0.1,
            sensor_specifications=sensor_spec,
            action_space=action_space,
            body_type="cylinder",
        )

    def move_and_look(self, action_name, steps=1):
        """
        Calls the given action on the agent a number of times and steps the simulation.
        """
        for _ in range(steps):
            self.agent.act(action_name)
            self.sim.step_world(1.0 / self.fps)

    def save_viewpoint_image(self, file_path):
        """
        Captures the current image from the agent's "color_sensor" render target and saves it.
        """
        # Ensure the latest view is rendered.
        self.draw_event()  

        # Access the sensor directly (using the internal simulator sensor dictionary).
        sensor = self.sim._Simulator__sensors[self.agent_id]["color_sensor"]
        # Read pixel data from the sensor's render target.
        # This returns a buffer of pixel data in RGBA format.
        image_buffer = sensor.render_target.read(mn.PixelFormat.RGBA, mn.PixelType.UNSIGNED_BYTE)
        
        # Convert the raw data into a NumPy array.
        # The shape is (height * width * 4); we need to reshape it to (height, width, 4)
        width = int(self.sim_settings["width"])
        height = int(self.sim_settings["height"])
        observation = np.frombuffer(image_buffer, dtype=np.uint8).reshape((height, width, 4))
        try:
            import imageio
            imageio.imwrite(file_path, observation)
            print(f"Saved viewpoint image to {file_path}")
        except ImportError:
            print("Please install imageio to save images (pip install imageio).")

    def draw_event(self):
        # Process any pending commands from the command queue.
        try:
            while True:
                action_name, steps = command_queue.get_nowait()
                print(f"Processing command: {action_name} {steps}")
                self.move_and_look(action_name, steps)
        except queue.Empty:
            pass

        # Clear framebuffer.
        mn.gl.default_framebuffer.clear(mn.gl.FramebufferClear.COLOR | mn.gl.FramebufferClear.DEPTH)
        # Render sensor observation to off-screen buffer and then blit it.
        self.sim._Simulator__sensors[self.agent_id]["color_sensor"].draw_observation()
        self.render_camera.render_target.blit_rgba_to_default()

        # Draw overlay text.
        self.shader.bind_vector_texture(self.glyph_cache.texture)
        self.shader.transformation_projection_matrix = self.window_text_transform
        self.shader.color = [1.0, 1.0, 1.0]
        self.window_text.render("Habitat-Sim Non-Interactive Viewer")
        self.shader.draw(self.window_text.mesh)

        self.swap_buffers()
        self.redraw()

def command_thread_func():
    """Simulate command input by enqueuing actions periodically."""
    while True:
        # For demonstration, every 2 seconds enqueue a move_forward command.
        time.sleep(0.1)
        # The command tuple: (action_name, steps)
        command_queue.put(("move_forward", 1))
        # You can add additional commands as needed.
        command_queue.put(("turn_right", 1))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="/Users/duanjs7/Desktop/habitat-sim/data/versioned_data/mp3d_example_scene_1.1/17DRP5sb8fy/17DRP5sb8fy.glb", type=str)
    parser.add_argument("--width", default=800, type=int)
    parser.add_argument("--height", default=600, type=int)
    args = parser.parse_args()

    sim_settings = default_sim_settings.copy()
    sim_settings["scene"] = args.scene
    sim_settings["window_width"] = args.width
    sim_settings["window_height"] = args.height
    sim_settings["default_agent"] = 0

    # Instantiate the viewer.
    # viewer = HabitatSimNonInteractiveViewer(sim_settings)
    
    # Instantiate the viewer.
    viewer = HabitatSimNonInteractiveViewer(sim_settings)

    # Start the command thread.
    cmd_thread = threading.Thread(target=command_thread_func, daemon=True)
    cmd_thread.start()

    # Start the application event loop (runs on the main thread).
    viewer.exec()
   

    

    