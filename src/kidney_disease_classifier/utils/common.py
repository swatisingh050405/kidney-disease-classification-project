import os
from box.exceptions import BoxValueError
from box import ConfigBox
import yaml
from pathlib import Path
from kidney_disease_classifier import logger
import json
import joblib
from ensure import ensure_annotations
from typing import Any
import base64


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a yaml file and returns 
    
    args:
        path_to_yaml: Path to the yaml file

    raises:
        BoxValueError: If there is an error while parsing the yaml file
        Exception: If there is any other exception

    returns:
        ConfigBox: ConfigBox object containing the yaml file content
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError as e:
        raise ValueError(f"Error while parsing yaml file: {e}")
    except Exception as e:
        raise e
    

@ensure_annotations
def create_directories(path_to_directories: list , verbose = True):
    """
    Creates a list of directories

    args:
        path_to_directories: List of paths to the directories to be created
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory created at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Saves a dictionary to a json file

    args:
        path: Path to the json file
        data: Dictionary to be saved
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"json file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> dict:
    """
    Loads a json file and returns a dictionary

    args:
        path: Path to the json file
    returns:
        dict: Dictionary containing the json file content   
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded from: {path}")
    return content


@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    Saves a binary file

    args:
        data: Data to be saved


    """   
    joblib.dump(value=data, path=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Loads a binary file and returns the content

    args:   

        path: Path to the binary file to be loaded
    returns:
        Any: Content of the binary file 

    """
    content = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return content


@ensure_annotations
def get_size(path: Path) -> str:
    """
    Returns the size of a file in KB

    args:
        path: Path to the file whose size is to be calculated
    returns:
        str: Size of the file in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024, 2)
    return f"{size_in_kb} KB"

def decodeImage(imgstring , filename):
    imgdata = base64.b64decode(imgstring)
    with open(filename, 'wb') as f:
        f.write(imgdata)
        f.close()

def encodeImageIntoBase64(cropped_image_path):
    with open(cropped_image_path, 'rb') as f:
        image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')