'''
Autor: Biranavan Parameswaran
'''
import os
import shutil
from PIL import Image
import numpy as np
def including_background():
    print("best and worst incld background")
    
    parent_dir = 'n_version_output'
    new_dir = './best_worst_incl_background'

    iou_values = {'Pixelwise IoU': [], 'Weighted IoU': []}

    for subdir in os.listdir(parent_dir):
        confidence_file = os.path.join(parent_dir, subdir, f'confidence_{subdir}.txt')
        
        if os.path.isfile(confidence_file):
            with open(confidence_file, 'r') as file:
                lines = file.readlines()

            for line in lines:
                if 'Pixelwise IoU' in line or 'Weighted IoU' in line:
                    iou_type, value = line.strip().split(': ')
                    iou_values[iou_type].append((float(value), subdir))

    new_directories = {
        'Pixelwise IoU': {'best': '5_best_pixelwise', 'worst': '5_worst_pixelwise'},
        'Weighted IoU': {'best': '5_best_weighted', 'worst': '5_worst_weighted'}
    }

    for iou_type, directories in new_directories.items():
        # sort the IoU values in descending order (best values)
        
        iou_values[iou_type].sort(reverse=True)
        for value, subdir in iou_values[iou_type][:5]:
            src_dir = os.path.join(parent_dir, subdir)
            dst_dir = os.path.join(new_dir, directories['best'], subdir)
            shutil.copytree(src_dir, dst_dir)
            print(f"subdir: {subdir} with value {value} for {directories}")
        # sort the IoU values in ascending order (worst values)
    
        iou_values[iou_type].sort()
        for value, subdir in iou_values[iou_type][:5]:
            src_dir = os.path.join(parent_dir, subdir)
            dst_dir = os.path.join(new_dir, directories['worst'], subdir)
            shutil.copytree(src_dir, dst_dir)
            print(f"subdir: {subdir} with value {value} for {directories}")

# do the same for excluding background data
def excludingbackground():
    import os


    parent_dir = './n_version_output'
    new_dir = './best_worst_exclud_background'

    iou_values = {'Pixelwise IoU': [], 'Weighted IoU': []}

    for subdir in os.listdir(parent_dir):
        label_image_path = os.path.join(parent_dir, subdir, f'label_{subdir}.png')
        
        if os.path.isfile(label_image_path):
            image = Image.open(label_image_path).convert('L')
            image_array = np.array(image)

            if np.all(image_array == 0):
                continue
        
        confidence_file = os.path.join(parent_dir, subdir, f'confidence_{subdir}.txt')
        
        if os.path.isfile(confidence_file):
            with open(confidence_file, 'r') as file:
                lines = file.readlines()

            for line in lines:
                if 'Pixelwise IoU' in line or 'Weighted IoU' in line:
                    iou_type, value = line.strip().split(': ')
                    iou_values[iou_type].append((float(value), subdir))

    new_directories = {
        'Pixelwise IoU': {'best': '5_best_pixelwise', 'worst': '5_worst_pixelwise'},
        'Weighted IoU': {'best': '5_best_weighted', 'worst': '5_worst_weighted'}
    }

    for iou_type, directories in new_directories.items():
        print("best and worst excluding background")
        iou_values[iou_type].sort(reverse=True)
        for value, subdir in iou_values[iou_type][:5]:
            src_dir = os.path.join(parent_dir, subdir)
            dst_dir = os.path.join(new_dir, directories['best'], subdir)
            shutil.copytree(src_dir, dst_dir)
            print(f"subdir: {subdir} with value {value} for {directories}")
        
        iou_values[iou_type].sort()
        for value, subdir in iou_values[iou_type][:5]:
            src_dir = os.path.join(parent_dir, subdir)
            dst_dir = os.path.join(new_dir, directories['worst'], subdir)
            shutil.copytree(src_dir, dst_dir)
            print(f"subdir: {subdir} with value {value} for {directories}")


if __name__ == '__main__':
    including_background()
    excludingbackground()