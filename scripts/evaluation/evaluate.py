'''
Autor: Biranavan Parameswaran
'''
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool

from tqdm import tqdm
import matplotlib.patches as mpatches
import json


CWD = os.getcwd()
plot_output = f"{CWD}/plots"

if not os.path.exists(plot_output):
    os.makedirs(plot_output)

def calculate_misclassification_rate(pred_mask, gt_mask):

    pred_mask = np.where(pred_mask == 255, 1, 0)
    gt_mask = np.where(gt_mask == 255, 1, 0)

    # compute number of incorrectly classified pixels
    incorrect_pixels = np.sum(pred_mask != gt_mask)

    # compute total number of pixels
    total_pixels = gt_mask.size

    # compute misclassification rate
    misclassification_rate = incorrect_pixels / total_pixels

    return misclassification_rate


def plot_missclasification(json_data, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    sum_pixel_error = {'VGG16-Unet': 0, 'MobileNet-SegNet': 0, 'WCID': 0,
                            'max pixelweise\n Kombination': 0, 'gewichtete Kombination': 0}

    for image_data in json_data.values():
        for model in sum_pixel_error.keys():
            sum_pixel_error[model] += image_data[model]

    mean_pixel_error = {model: total / len(json_data) for model, total in sum_pixel_error.items()}

    group1 = ['VGG16-Unet', 'MobileNet-SegNet', 'WCID']
    group2 = [ 'max pixelweise\n Kombination','gewichtete Kombination']

    colors = ['#1f77b4', '#ff7f0e']

    blue_patch = mpatches.Patch(color=colors[0], label='Einzelnes Modell')
    red_patch = mpatches.Patch(color=colors[1], label='N-Version System')
    
    plt.figure(figsize=(10, 6))

    bars_group1 = plt.bar(group1, [mean_pixel_error[model] for model in group1], color=colors[0])

    bars_group2 = plt.bar(group2, [mean_pixel_error[model] for model in group2], color=colors[1])

    for bars in [bars_group1, bars_group2]:
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval / 2,
                     round(yval, 4), ha='center', va='center', color='white')

    plt.xlabel('Modell', labelpad=10)
    plt.ylabel(f'durschnittliche Fehlerrate in Pixel')
    plt.title(f'durschnittliche Fehlerrate pro Modell')

    plt.legend(handles=[blue_patch, red_patch])

    plt.savefig(os.path.join(save_dir, "missclasification.png"))
    plt.close()



    
def plot_mean_difference(data):

    df_group1 = pd.DataFrame(columns=['MobileNet-SegNet', 'VGG16-Unet', 'WCID'])
    df_group2 = pd.DataFrame(columns=['max pixelweise\n Kombination', 'gewichtete Kombination'])

    for key in data:
        df_group1.loc[key] = [data[key]['MobileNet-SegNet']['difference'], data[key]['VGG16-Unet']['difference'], data[key]['WCID']['difference']]
        df_group2.loc[key] = [data[key]['max pixelweise\n Kombination']['difference'], data[key]['gewichtete Kombination']['difference']]

    mean_group1 = df_group1.mean()
    mean_group2 = df_group2.mean()

    mean_combined = pd.concat([mean_group1, mean_group2])

    colors = ['#1f77b4'] * len(mean_group1) + ['#ff7f0e'] * len(mean_group2)

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(mean_combined.index, mean_combined.values, color=colors)
    ax.set_title('Durschnittliche Abweichung der weitesten erkannten Schiene von der ground truth')
    ax.set_ylabel('durschnittlicher Betrag der Differenz in Pixeln')
    ax.set_xlabel('Modelle', labelpad=15)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 3), va='bottom', ha='center')


    blue_patch = mpatches.Patch(color='#1f77b4', label='Einzelnes Modell')
    orange_patch = mpatches.Patch(color='#ff7f0e', label='N-Version System')
    plt.legend(handles=[blue_patch, orange_patch])

    plt.tight_layout()
    plt.savefig(f'{plot_output}/mean_difference.png')


def find_smallest_y(img, kernel_size=5):
    smallest_y = -1
    # define kernel size for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # perform operation to remove noise (erosion followed by dilation)
    mask = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    for y in reversed(range(mask.shape[0])):
        if np.any(mask[y, :] == 255):  
            smallest_y = min(smallest_y, y) if smallest_y != -1 else y 
    
    
    return smallest_y





def calculate_midpoints(mask):
    mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    midpoints = {}

    for contour in contours:
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

        # iterate over each row (y level)
        for y, row in enumerate(contour_mask):
            white_pixels = np.where(row == 255)[0]
            if white_pixels.size != 0:  
                # find the leftmost and rightmost white pixels
                left = np.min(white_pixels)
                right = np.max(white_pixels)
                # calculate the midpoint
                midpoint_x = (left + right) / 2
                if y not in midpoints:
                    midpoints[y] = []
                midpoints[y] = midpoints[y] + [midpoint_x]
                
    return midpoints



def load_image(img_dir):
    # build paths
    output_dir = os.path.join(img_dir, "individual_output")
    evaluator_dir = os.path.join(img_dir, "evaluator")

    label_path = os.path.join(img_dir, f"label_{os.path.basename(img_dir)}.png")
    segnet_path = os.path.join(output_dir, "segnet_output.png")
    unet_path = os.path.join(output_dir, "unet_output.png")
    wcid_path = os.path.join(output_dir, "wcid_output.png")
    pixel_wise_path = os.path.join(evaluator_dir, "pixel_wise_output.png")
    weighted_voting_path = os.path.join(evaluator_dir, "weighted_average_output.png")

    data_paths = {"label": label_path, "MobileNet-SegNet": segnet_path, "VGG16-Unet": unet_path, "WCID": wcid_path,
                  "max pixelweise\n Kombination": pixel_wise_path,
                  "gewichtete Kombination": weighted_voting_path
                  }

    return os.path.basename(img_dir), {k: cv2.imread(v, cv2.IMREAD_GRAYSCALE) for k, v in data_paths.items() if os.path.isfile(v)}

def load_images_parallel(path: str) -> np.ndarray:
    # load images parallel
    predictions = [os.path.join(path, img_name) for img_name in os.listdir(path) if os.path.isdir(os.path.join(path, img_name))]
    
    with Pool(os.cpu_count()) as p:
        data = dict(tqdm(p.imap(load_image, predictions), total=len(predictions), desc="Loading images"))

    return data




def closest_value(x, x_list):
    # find the closest value in x_list to x
    if len(x_list) == 0: 
        return None
    return x_list[np.argmin(np.abs(np.array(x_list) - x))]

def distances_per_y(gt_midpoints, pred_midpoints):
    # calculate midpoint distances per y
    distances = {}
    for y,x_list in gt_midpoints.items():
        x_pred_list = pred_midpoints.get(y, []) 
        distances[y] = {
            "error": [],
            "difference": abs(len(x_pred_list)-len(x_list))
            }
        if len(x_pred_list) == 0:
            continue
            
            
       
        for x in x_list:
            x_pred = closest_value(x, x_pred_list)
            if x_pred is not None:
                saved_errors = distances[y]["error"]
                saved_errors.append(abs(x - x_pred))
                distances[y]["error"] = saved_errors
                x_pred_list.remove(x_pred)


    return distances            
     

            
           


class ImageEvaluator:
    def __init__(self, dataset_path: str):
        self.data = load_images_parallel(dataset_path)
     
        self.evaluate()

    def evaluate(self):
        self.calculate_height_diff()
        self.calculate_background_images()
        self.calculate_and_plot_midpoints()
        

    
    def calculate_and_plot_midpoints(self):
            # check if dir exists else create
            midpoint_path = os.path.join(plot_output,"midpoint")
            os.makedirs(os.path.join(midpoint_path), exist_ok=True)
            
            midpoints = {}
            gt_midpoints = {}
            data = self.data.copy()
            for prediction_name, models in tqdm(data.items(), desc="Calculating midpoints error"):
                models = models.copy()

                midpoints[prediction_name] = {}
                true_midpoints = calculate_midpoints(models["label"])   
                gt_midpoints[prediction_name] = true_midpoints
                    
                                
                models.pop("label")
                for model_name, prediction in models.items():
                    predicted_midpoints = calculate_midpoints(prediction)
                    prediction = cv2.cvtColor(prediction, cv2.COLOR_GRAY2RGB)
                    midpoints[prediction_name][model_name] = distances_per_y(true_midpoints, predicted_midpoints)
            
            # summarize y levels for total midpoints calculation
            y_summary = {}
            for img_name,y_error_dict in gt_midpoints.items():
                for y_level, metrics in y_error_dict.items():
                    if y_level not in y_summary:
                        y_summary[y_level] = 0
                    y_summary[y_level] = y_summary[y_level] + len(metrics)
            y_summary = {key:y_summary[key] for key in sorted(y_summary.keys())}
            
            
            
            # initialize lists to hold data
            img_names = []
            model_names = []
            y_levels = []
            errors = []
            differences = []

            for img_name, models in midpoints.items():
                for model_name, y_data in models.items():
                    for y_level, metrics in y_data.items():
                        img_names.append(img_name)
                        model_names.append(model_name)
                        y_levels.append(int(y_level))  
                        errors.append(metrics['error'])
                        differences.append(metrics['difference'])

            df = pd.DataFrame({
                'img_name': img_names,
                'model_name': model_names,
                'y_level': y_levels,
                'error': errors,
                'difference': differences
            })

            # for each y_level in each model, calculate the mean error
            df['mean_error'] = df['error'].apply(lambda x: sum(x) / len(x) if x else None)

            # compute the mean error per y-level per model
            mean_error_per_y_level = df.groupby(['model_name', 'y_level'])['mean_error'].mean().reset_index()

            # create plot
            plt.figure(figsize=(10, 6))
            for model, data in mean_error_per_y_level.groupby('model_name'):
                plt.plot(data['y_level'], data['mean_error'], label=model)
            plt.xlabel('Y-Level')
            plt.ylabel('Durschnittlicher Betrag des Fehlers in Pixeln')
            plt.title('Durschnittlicher Mittelpunkt Fehler pro Modell')
            plt.legend()
            plt.savefig(f'{midpoint_path}/midpoints.png')

            # calculate the mean differecne per y level per model
            df['individual_error_count'] = df['error'].apply(len)



            
            
            order = ['VGG16-Unet', 'MobileNet-SegNet', 'WCID', 'max pixelweise\n Kombination', 'gewichtete Kombination']
            colors = ['#1f77b4' if model in ['VGG16-Unet', 'MobileNet-SegNet', 'WCID'] else '#ff7f0e' for model in order]
            labels = ['Einzelnes Modell' if model in ['VGG16-Unet', 'MobileNet-SegNet', 'WCID'] else 'N-Version System' for model in order]

            plt.figure(figsize=[10,6])
            mean_difference_per_model = df.groupby('model_name')['difference'].mean().reset_index()


            unique_labels_colors = list(set(zip(labels, colors)))

            legend_patches = [mpatches.Patch(color=color, label=label) for label, color in unique_labels_colors]

            for idx, row in mean_difference_per_model.iterrows():
                bar = plt.bar(row['model_name'], row['difference'], color=colors[idx])
                
                plt.text(bar[0].get_x() + bar[0].get_width()/2.0, 
                        bar[0].get_height(), 
                        round(row['difference'], 3), 
                        ha='center', 
                        va='bottom')

            plt.title('Durschnittliche Abweichung der Mittelpunkte pro Modell')
            plt.xlabel('Model', labelpad=10)
            plt.ylabel('durschnittliche Abweichung pro Y-Level in Pixeln', labelpad=10)
        

            plt.legend(handles=legend_patches)

            plt.savefig(f'{midpoint_path}/mean_difference_per_model.png')
            mean_errors = {}

            # Iterate over each prediction in the data and calculate the mean error for each model
            data = midpoints.copy()
            for prediction_name, models in data.items():
                for model_name, midpoints in models.items():
                    all_errors = []
                    for y, info in midpoints.items():
                        all_errors += info['error']
                    mean_error = sum(all_errors) / len(all_errors) if all_errors else 0
                    if model_name not in mean_errors:
                        mean_errors[model_name] = []
                    mean_errors[model_name].append(mean_error)

            for model_name, errors in mean_errors.items():
                mean_errors[model_name] = sum(errors) / len(errors)
            colors = ['#1f77b4' if x in ['VGG16-Unet', 'MobileNet-SegNet', 'WCID'] else '#ff7f0e' for x in mean_errors.keys()]
            labels = ['Einzelnes Modell' if x in ['VGG16-Unet', 'MobileNet-SegNet', 'WCID'] else 'N-Version System' for x in mean_errors.keys()]

            plt.figure(figsize=[10,6])

            bars = plt.bar(mean_errors.keys(), mean_errors.values(), color=colors)

            handles = [plt.Rectangle((0,0),1,1, color=color) for color in set(colors)]
            labels = ['Einzelnes Modell' if color=='#1f77b4' else 'N-Version System' for color in set(colors)]
            plt.legend(handles, labels)

            plt.xlabel('Modell', labelpad=10)
            plt.ylabel('Durschnittlicher Fehler', labelpad=10)
            plt.title('Durschnittlicher Fehler pro Modell', pad=10)
            plt.show()
            plt.savefig(os.path.join(midpoint_path, "mean_error_bar.png"))
            
            
           
            # plot the total midpoints for each y-level
                      
            x_values = list(y_summary.keys())
            y_values = list(y_summary.values())


            # sort 
            sorted_pairs = sorted(zip(x_values, y_values))

            # unzip 
            x_values, y_values = zip(*sorted_pairs)

            # plot
            plt.figure(figsize=(10, 6))
            plt.plot(x_values, y_values, "-", color='tab:blue', label='Summe der Mittelpunkte')
            plt.xlabel('Y-Level')
            plt.ylabel('Summe der Mittelpunkte')
            plt.title('Summe der Mittelpunkte pro Y-Level (alle Testbilder)')


            plt.legend()
            plt.show()
            plt.savefig(f'{midpoint_path}/total_midpoints.png')
               
    def calculate_background_images(self):
        background_images = {}



        for image_name, models in tqdm(self.data.items(), desc="Calculating background images"):
            models = models.copy()  
            label = models["label"]
            models.pop("label")
            check = np.all(label== 0)
            # if there are no white pixels in the label, calculate the misclassification rate for each model
            if(check):
                background_images[image_name] = {}
                
                for model_name, prediction in models.items():
                    background_images[image_name][model_name] = calculate_misclassification_rate(label, prediction)
        # plot the misclassification rate for each model
        plot_missclasification(background_images, os.path.join(plot_output, "background_images"))
   
    def calculate_height_diff(self):
        # calculate the height (y min) difference for each model
        height_diff = {}
        data = self.data.copy()
        for prediction_name,models in tqdm(data.items(), desc="Calculating height difference"):
            models = models.copy() 

            height_diff[prediction_name] = {}
            label_height = find_smallest_y(models["label"])
            height_diff[prediction_name]["label"] = label_height
            models.pop("label")
            for model_name, prediction in models.items():
                y = find_smallest_y(prediction)
                # add difference  from label for each model
                height_diff[prediction_name][model_name] ={
                    "height": y,
                    "difference": abs( y - label_height)
                }
        # plot data
        plot_mean_difference(height_diff)
    
if __name__ == "__main__":
    dataset_path ="/n_version_output"
    ImageEvaluator(dataset_path)

