import os
import time
import pandas as pd

from config_util import (
    MP3D_DATASET_PATH,
    DEFAULT_GML_SAVE_PATH,
    DEFAULT_IMG_SAVE_PATH,
    MP3D_DATASET_SCENE_IDS_LIST
)
from adaptive_topo.construct_graph import GraphConstructor
from vision_gd import ObjectExtractor

if __name__ == "__main__":
    timings = []
    # Initialize the object extractor
    myOE =  ObjectExtractor(threshold=0.9)  # Initialize the object extractor
    for scene_id in sorted(MP3D_DATASET_SCENE_IDS_LIST):
        print(f"\n=== Processing scene: {scene_id} ===")
        start_time = time.time()

        # try:
            # Instantiate graph constructor
        gcb = GraphConstructor(
            scene_id=scene_id,
            save_gml_path=DEFAULT_GML_SAVE_PATH,
            save_img_path=DEFAULT_IMG_SAVE_PATH,
            is_level_derive=True,
            save_graph=True,
            show_graph=False,
            object_extractor=myOE  
        )

        # Build (and optionally save) the topological graph
        gcb.construct_topological_graph()

        # Clean up simulator
        gcb.sim.close()

        # except Exception as e:
        #     print(f"[Error] Failed to process scene {scene_id}: {e}")

        elapsed = time.time() - start_time
        timings.append({
            "scene_id": scene_id,
            "time_sec": elapsed
        })
        break

    # Summary of processing times
    print("\n=== Scene Processing Time Summary ===")
    df = pd.DataFrame(timings)
    print(df.to_string(index=False))

    total_time = df["time_sec"].sum()
    print(f"\n=== Total Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes) ===")
