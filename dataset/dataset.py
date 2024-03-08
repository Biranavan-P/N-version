import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
'''
Autor: Biranavan Parameswaran
'''
class Dataset:
    def __init__(self, train_img: str, train_annotation: str, valid_img: str, valid_annotation: str,
                 test_img:str = None,test_annotation:str=None):
        self.train_img = (train_img)
        self.train_annotation = (train_annotation)
        self.valid_img = (valid_img)
        self.valid_annotation = (valid_annotation)
        self.test_img = (test_img)
        self.test_annotation = (test_annotation)

    def count_background_images(self, annotation_folder: Path) -> int:
        # background images are images where the annotation is black (no target class)
        background_count = 0
        for img_file in (Path(annotation_folder).iterdir()):
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if np.all(img == 0):
                background_count += 1
        return background_count

    def summary(self):
        # print summary of dataset
        total_train_images = len(list(Path(self.train_img).iterdir()))
        total_valid_images = len(list(Path(self.valid_img).iterdir()))
        total_test_images = len(list(Path(self.test_img).iterdir())) if self.test_img is not None else 0
        total_images = total_train_images + total_valid_images + total_test_images

        background_train = self.count_background_images(self.train_annotation)
        background_valid = self.count_background_images(self.valid_annotation)
        background_test = self.count_background_images(self.test_annotation) if self.test_annotation is not None else 0
        background_total = background_train + background_valid + background_test

        perc_train = (total_train_images / total_images) * 100
        perc_valid = (total_valid_images / total_images) * 100
        perc_background_total = (background_total / total_images) * 100  if background_total > 0 else 0
        perc_background_train = (background_train / total_train_images) * 100 
        perc_background_valid = (background_valid / total_valid_images) * 100

        
        print(f"Total number of images: {total_images}")
        print(f"Total number of images in train_img: {total_train_images}")
        print(f"Total number of images in valid_img: {total_valid_images}")
        print(f"Percentage of images in train_img: {perc_train:.2f}%")
        print(f"Percentage of images in valid_img: {perc_valid:.2f}%")
        
        
        print(f"Total number of background images: {background_total}")
        print(f"Percentage of background images in total: {perc_background_total:.2f}%")
        print(f"Percentage of background images in train: {perc_background_train:.2f}%")
        print(f"Percentage of background images in valid: {perc_background_valid:.2f}%")
        # if testset is available
        if self.test_annotation is not None:
            perc_test = (total_test_images / total_images) * 100

            perc_background_test = (background_test / total_test_images) * 100

            print("data testset")
            print(f"Percentage of images in test_img: {perc_test:.2f}%")

            print(f"Total number of images in test_img: {total_test_images}")
            print(f"Percentage of background images in test: {perc_background_test:.2f}%")


if __name__ == '__main__':
    save_path="./data"
    train_img = save_path + "/images_train"
    annotation_img = save_path + "/annotations_train"
    val_img = save_path + "/images_valid"
    val_annotation_img = save_path + "/annotations_valid"
    test_img = save_path + "/images_test"
    test_annotation_img = save_path + "/annotations_test"
    dataset = Dataset(train_img, annotation_img, val_img, val_annotation_img,test_img,test_annotation_img)
    dataset.summary()