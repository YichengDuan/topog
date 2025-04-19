import os
import time
import pandas as pd
from joblib import Parallel, delayed

from config_util import (
    MP3D_DATASET_PATH,
    DEFAULT_GML_SAVE_PATH,
    DEFAULT_IMG_SAVE_PATH,
    MP3D_DATASET_SCENE_IDS_LIST
)
from adaptive_topo.construct_graph import GraphConstructor
from vision_gd import ObjectExtractor
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

def process_scene(scene_id, gml_root, img_root, threshold):
    """
    Instantiate the GraphConstructor + ObjectExtractor, run it, and
    return a dict with timing info.
    """
    start = time.time()

    # Each worker gets its own extractor (to avoid cross‑process state issues)
    extractor = ObjectExtractor(threshold=threshold)

    gc = GraphConstructor(
        scene_id=scene_id,
        save_gml_path=gml_root,
        save_img_path=img_root,
        is_level_derive=True,
        save_graph=True,
        show_graph=False,
        object_extractor=extractor
    )
    gc.construct_topological_graph()
    gc.sim.close()

    return {
        "scene_id": scene_id,
        "time_sec": time.time() - start
    }

if __name__ == "__main__":
    # prepare save roots
    gml_root = DEFAULT_GML_SAVE_PATH
    img_root = DEFAULT_IMG_SAVE_PATH
    threshold = 0.9

    # Number of parallel jobs: -1 means “use all CPUs”
    results = Parallel(n_jobs=5)(
        delayed(process_scene)(
            scene_id,
            gml_root,
            img_root,
            threshold
        )
        for scene_id in sorted(MP3D_DATASET_SCENE_IDS_LIST)
    )

    # Summarize
    df = pd.DataFrame(results)
    print("\n=== Scene Processing Time Summary ===")
    print(df.to_string(index=False))

    total = df["time_sec"].sum()
    print(f"\n=== Total Time: {total:.2f} seconds ({total/60:.2f} minutes) ===")