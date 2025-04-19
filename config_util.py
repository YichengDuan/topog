import yaml
import os
CONFIG_FILE_PATH = './local.yaml'


def load_config(file_path):
    """
    Load the configuration file.
    :param file_path: Path to the configuration file.
    :return: Configuration dictionary.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config(CONFIG_FILE_PATH)

MP3D_DATASET_PATH = config['mp3d_habitat_scene_dataset_path']
DEFAULT_GML_SAVE_PATH = config['default_gml_save_path']
DEFAULT_IMG_SAVE_PATH = config['default_img_save_path']
DEFAULT_DATASETS_PATH = config['datasets_path']
os.makedirs(DEFAULT_DATASETS_PATH, exist_ok=True)
os.makedirs(DEFAULT_DATASETS_PATH+'raw/', exist_ok=True)
os.makedirs(DEFAULT_DATASETS_PATH+'processed/', exist_ok=True)


def get_scene_list(MP3D_DATASET_PATH:str):
    """
    Read sub level scene list from folder.
    :param MP3D_DATASET_PATH: Path to the MP3D dataset.
    :return: List of scene names.
    """
    
    scene_ids_list = [d for d in os.listdir(MP3D_DATASET_PATH)
                 if os.path.isdir(os.path.join(MP3D_DATASET_PATH, d))
                 and os.path.exists(os.path.join(MP3D_DATASET_PATH, d, f"{d}.glb"))]
    

    return scene_ids_list

MP3D_DATASET_SCENE_IDS_LIST = get_scene_list(MP3D_DATASET_PATH)

print(MP3D_DATASET_SCENE_IDS_LIST)