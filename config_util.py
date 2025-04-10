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

def get_scene_list(MP3D_DATASET_PATH:str):
    """
    Read sub level scene list from folder.
    :param MP3D_DATASET_PATH: Path to the MP3D dataset.
    :return: List of scene names.
    """
    scene_list = []
    for root, dirs, files in os.walk(MP3D_DATASET_PATH):
        for dir in dirs:
            scene_list.append(dir)

    return scene_list

MP3D_DATASET_SCENE_LIST = get_scene_list(MP3D_DATASET_PATH)

