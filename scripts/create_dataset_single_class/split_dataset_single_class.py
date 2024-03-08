"""

Autor: Biranavan Parameswaran

This Dataset splits the Railsem19 dataset into train , validation and dataset.
Following Railway classes are used:
12: rail-track 
17: rail-raised 

The class numbers represent the pixel value in the uint8 encoded mask.
This script changes the pixel values 12 and  17 to 1.
The pixel values of 12 and 17 are read from the json file.


All other classes are set to 0, resulting to following classes:
0: background
1: rail-track
1: rail-raised
The overall ratio is 70% train, 20% validation and 10% test.

example usage:
python split_dataset.py --save_path ./data --dataset /path_to_dataset/railsem

Please use it on the original dataset.
"""
import shutil
import argparse
import math
import random
from pathlib import Path
import copy
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import os

def save_mask(path:str,msk:np.ndarray):
    cv2.imwrite(path,msk)
def save_symlink(old_path,new_path):
    os.symlink(old_path,new_path)

def normal_round(n):
    if n - math.floor(n) < 0.5:
        return math.floor(n)
    return math.ceil(n)

def get_class_index(json_file: str, class_name: str) -> int:
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    for i, label in enumerate(data['labels']):
        if label['name'] == class_name:
            return i
    
    raise ValueError(f"Class {class_name} not found in {json_file}")
def validate_mask(mask_original: np.ndarray, mask_changed: np.ndarray, rail_track_index: int, rail_raised_index: int) -> bool:
    """
    Validate if the mask was changed correctly.
    """

    # check if the masks have the same shape
    if mask_original.shape != mask_changed.shape:
        msg = ("The masks do not have the same shape.")
        return False,msg

    # get all the indices where mask_original has value rail_track_index
    rail_track_indices = np.where(mask_original == rail_track_index)

    # get all the indices where mask_original has value rail_raised_index
    rail_raised_indices = np.where(mask_original == rail_raised_index)

    # get all the indices where mask_original has neither rail_track_index nor rail_raised_index
    other_indices = np.where((mask_original != rail_track_index) & (mask_original != rail_raised_index))

    # check if all these indices have the correct values in mask_changed
    if not np.all(mask_changed[rail_track_indices] == 1):
        msg = ("Some rail track indices are not correctly mapped in the changed mask.")
        mask_changed[mask_changed==1] = 255
        mask_changed[mask_changed==1] = 125
        mask_original[mask_original!=rail_track_index & mask_original!=rail_raised_index] = 0

        mask_original[mask_original==rail_track_index] = 255
        mask_original[mask_original==rail_raised_index] = 125
        cv2.imwrite("mask_changed_track.png",np.hstack((mask_changed,mask_original)))
        
        return False,msg

    if not np.all(mask_changed[rail_raised_indices] == 1):
        msg = ("Some rail raised indices are not correctly mapped in the changed mask.")
        mask_changed[mask_changed==1] = 255
        mask_changed[mask_changed==1] = 125
        mask_original[mask_original!=rail_track_index & mask_original!=rail_raised_index] = 0
        mask_original[mask_original==rail_track_index] = 255
        mask_original[mask_original==rail_raised_index] = 125
        cv2.imshow("mask_changed_raised.png",np.hstack((mask_changed,mask_original)))
        
        return False,msg

    if not np.all(mask_changed[other_indices] == 0):
        msg = ("Some other indices are not correctly mapped in the changed mask.")
        mask_changed[mask_changed==0] = 255

        mask_original[mask_original==rail_track_index] = 255
        mask_original[mask_original==rail_raised_index] = 125
        cv2.imshow("mask_changed_other.png",np.hstack((mask_changed,mask_original)))
        return False,msg

    # if all checks passed, return True
    return True,"passed"

def split_data(save_path:str, dataset:str,train_size_perc=0.7,valid_size_perc=0.2,test_size_perc=0.1,test_size_bg=0.05):
    # create temp dir
    if (save_path=="./tmp") or dataset=="./tmp":
        raise Exception("Please dont use ./tmp as path")
    tmp_path = Path("./tmp")
    bg_path = (tmp_path / "bg")
    non_bg_path = (tmp_path/"non_bg")
        
    
    if not tmp_path.exists():
        tmp_path.mkdir()
        
    if not bg_path.exists():
        bg_path.mkdir()
    if not non_bg_path.exists():
        non_bg_path.mkdir()
            
    
    
        
    # check and create folders
    
    save_path = Path(save_path)  
    if not save_path.exists():
        save_path.mkdir()
    for subset in ["train", "valid", "test"]:
        if not (save_path / f"images_{subset}").exists():
            (save_path / f"images_{subset}").mkdir()
        if not (save_path / f"annotations_{subset}").exists():
            (save_path / f"annotations_{subset}").mkdir()
    img_path =dataset+ "/jpgs/rs19_val"
    json_path = dataset + "/rs19-config.json"
    images = list(Path(img_path).iterdir())
   
    # get index of rail-track and rail-raised classes
    rail_track_index = get_class_index(json_path, "rail-track")
    rail_raised_index = get_class_index(json_path, "rail-raised")
    print(f"rail-track index: {rail_track_index}")
    print(f"rail-raised index: {rail_raised_index}")
    img_non_bg_data = {
        
    }
    img_bg_data= {
        
    }
    
    for img in tqdm(images, desc="Processing images and validating changed mask"):
        msk_path = str(img).replace("jpgs", "uint8").replace(".jpg", ".png")
        msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        msk_original = copy.deepcopy(msk)
        assert np.array_equal(msk, msk_original)
        tmp_rail_track_indx = 222
        tmp_rail_raised_indx = 223

        # convert rail-track and rail-raised pixels to temporary indices
        msk[msk == rail_track_index] = tmp_rail_track_indx
        msk[msk == rail_raised_index] = tmp_rail_raised_indx

        # convert all remaining pixels to 0
        msk[(msk != tmp_rail_track_indx) & (msk != tmp_rail_raised_indx)] = 0
        key = img.stem
        
        # check if msk is a background image then save it to bg folder
        if np.all(msk == 0):
            verified,msg = validate_mask(mask_changed=msk, mask_original=msk_original,
                                         rail_raised_index=rail_raised_index, rail_track_index=rail_track_index)    
            if not verified:
                    raise ValueError(f"Image {img} failed validation. Reason: {msg}")
            save_mask_path = str(bg_path.joinpath(f"{img.name}.png"))

            save_mask(save_mask_path,msk)  
            img_bg_data[key] = {
                "msk_path":save_mask_path,
                "img_path": str(img)
            }
        else:
            
            # give the rail-raised and rail-track pixels the correct index
            msk[msk == tmp_rail_track_indx] = 1
            msk[msk == tmp_rail_raised_indx] = 1
            # check if the changes are correct
            verified,msg = validate_mask(mask_changed=msk, mask_original=msk_original,
                                         
                                         rail_raised_index=rail_raised_index, rail_track_index=rail_track_index)    
            # if not correct then raise error
            if not verified:
                print(f"Image {img} failed validation. Reason: {msg}")
                raise ValueError(f"Image {img} failed validation. Reason: {msg}")
            # save the mask to non_bg folder
            save_mask_path = str(non_bg_path.joinpath(f"{img.name}.png"))
            save_mask(save_mask_path,msk)  

            img_non_bg_data[key] = {
                "msk_path":save_mask_path,
                "img_path": str(img)
            }
    
    images_bg = list(img_bg_data.keys())
    images_non_bg = list(img_non_bg_data.keys())
    
    # shuffle the images
    random.shuffle(images_bg)
    random.shuffle(images_non_bg)
    

    # total number of background images
    background_imgs_size = len(images_bg)

    # total number of non-background images
    non_background_imgs_size = len(images_non_bg)
    
    # total number of images
    total_imgs = background_imgs_size + non_background_imgs_size

    print(background_imgs_size)
    print(non_background_imgs_size)
    print(total_imgs)

    total_test_size = normal_round(total_imgs * test_size_perc)

    # calculate the number of background and non-background images in the test set
    back_ground_test_size = normal_round(background_imgs_size * 0.05)  # Around 5% of background images for testing
    non_back_ground_test_size = total_test_size - back_ground_test_size  # The rest of the test set will be non-background images

    # split the images
    test_bg = images_bg[:back_ground_test_size]
    images_bg = images_bg[back_ground_test_size:]

    test_img = images_non_bg[:non_back_ground_test_size]
    images_non_bg = images_non_bg[non_back_ground_test_size:]

    # calculate the sizes for training and validation sets
    background_train_size = normal_round(len(images_bg) * train_size_perc / (train_size_perc + valid_size_perc))

    non_background_train_size = normal_round(len(images_non_bg) * train_size_perc / (train_size_perc + valid_size_perc))

    # split the remaining images into training and validation sets
    train_bg, valid_bg = train_test_split(images_bg, train_size=background_train_size, random_state=42)
    train_img, valid_img = train_test_split(images_non_bg, train_size=non_background_train_size, random_state=42)
    
    
    

    # combine the splits
    train_img.extend(train_bg)
    valid_img.extend(valid_bg)
    test_img.extend(test_bg)
    
    
    # distribute the images to the respective folders and save	
    for subset,images in [("train",train_img),("valid",valid_img),("test",test_img)]:
        for img_name in tqdm(images, desc=f"saving {subset} data"):
            if img_name in img_bg_data:
               msk_path = img_bg_data[img_name]["msk_path"]
               img_path = img_bg_data[img_name]["img_path"]
            else:
                msk_path = img_non_bg_data[img_name]["msk_path"]
                img_path = img_non_bg_data[img_name]["img_path"]
                
            annotations_path = f"{save_path}/annotations_{subset}/{img_name}.png"
            img_save_path = f"{save_path}/images_{subset}/{img_name}.jpg"
            shutil.move(msk_path,annotations_path)
            os.symlink(img_path,img_save_path)
            
    # remove the temporary folder
    shutil.rmtree(str(tmp_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, help="Path to save the data")
    parser.add_argument("--dataset", type=str, help="Path to the Railsem dataset")
    
    args = parser.parse_args()
    
    save_path = (args.save_path)  
    dataset = (args.dataset)
    
    split_data(save_path, dataset)
   
    





