'''
Autor: Biranavan Parameswaran
'''
from pathlib import Path
from n_version import N_Version,calculate_iou
import argparse
import cv2
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
def plot_and_save_metrics(model_names, model_iou_values, model_confidence_values, save_path,title):
    assert len(model_names) == len(model_iou_values) == len(model_confidence_values), \
        "Mismatch in number of models and their IoU/Confidence values"

    # calculate average IoU and Confidence for each model
    avg_iou_values = [np.mean(iou_values) for iou_values in model_iou_values]
    avg_conf_values = [np.mean(conf_values) for conf_values in model_confidence_values]

    data = [avg_iou_values, avg_conf_values]

    bar_width = 0.35
    r1 = np.arange(len(avg_iou_values))
    r2 = [x + bar_width for x in r1]

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ['#1f77b4', '#ff7f0e']


    rects1 = ax.bar(r1, data[0], width=bar_width, color=colors[0], label='IoU')
    rects2 = ax.bar(r2, data[1], width=bar_width, color=colors[1], label='Konfidenzscore')

    ax.set_xlabel('Modell',labelpad=15)
    ax.set_ylabel('Wert', labelpad=15)  
    ax.set_title(title)
    ax.set_xticks([r + bar_width / 2 for r in range(len(avg_iou_values))])
    ax.set_xticklabels(model_names)
    ax.legend()

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., height,
                    '%.4f' % height, ha='center', va='bottom')
    ax.set_ylim([0, 1.15 * max(max(avg_iou_values), max(avg_conf_values))])

    plt.savefig(save_path)
    plt.close(fig)


def plot_and_save_sing_metrics(model_names, model_iou_values, model_confidence_values, save_path,title):
    assert len(model_names) == len(model_iou_values) == len(model_confidence_values), \
        "Mismatch in number of models and their IoU/Confidence values"

    data = [model_iou_values, model_confidence_values]

    bar_width = 0.35
    r1 = np.arange(len(model_iou_values))
    r2 = [x + bar_width for x in r1]

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ['#1f77b4', '#ff7f0e']

    rects1 = ax.bar(r1, data[0], width=bar_width, color=colors[0], label='IoU')
    rects2 = ax.bar(r2, data[1], width=bar_width, color=colors[1], label='Confidence')

    ax.set_xlabel('Model',labelpad=15)
    ax.set_ylabel('Scores', labelpad=15)  
    ax.set_title(title)
    ax.set_xticks([r + bar_width / 2 for r in range(len(model_iou_values))])
    ax.set_xticklabels(model_names)
    ax.legend()

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., height,
                    '%.4f' % height, ha='center', va='bottom')
    ax.set_ylim([0, 1.15 * max(max(model_iou_values), max(model_confidence_values))])
    
    plt.savefig(save_path)
    plt.close(fig)

    
def main( img_width, img_height, output_pth, image_path,model: N_Version):
    n_classes = 3
    img_list = (Path(f"{image_path}/images_test").iterdir())
    
    
    confidence_unet = []
    confidence_segnet = []
    confidence_wcid = []
    
    confidence_pixelwise = []
    confidence_weighted = []
    
    iou_unet = []
    iou_segnet = []
    ioU_wcid = []
    
    iou_pixelwise_ = []
    iou_weighted_ = []
    
    
    for img in tqdm(img_list):
        label_path = str(img).replace("images_test","annotations_test").replace(".jpg",".png")
        if not os.path.exists(label_path):
            print(label_path)
            raise ValueError("Label path does not exist.")
        label =  cv2.imread(label_path,0)
        label = np.where(label!=0,255,0)
        label = cv2.resize(label,(img_width,img_height),interpolation=cv2.INTER_NEAREST)
        try:
            output_dict = model.predict(n_classes, img_width, img_height, output_pth, img,save_img=True)
        except Exception as e:
            print(f"Error in prediction for {img}")
            print(e)
            continue
        # get metrics and output
        unet = output_dict["unet"]
        segnet = output_dict["segnet"]
        wcid = output_dict["wcid"]
        
        pixelwise = output_dict["pixelwise"]
        weighted = output_dict["weighted"]
        
        unet_output = unet[0]
        unet_confidence = unet[1]
        
        segnet_output = segnet[0]
        segnet_confidence = segnet[1]
        
        wcid_output = wcid[0]
        wcid_confidence = wcid[1]
        
        pixelwise_output = pixelwise[0]
        pixelwise_confidence = pixelwise[1]
        
        weighted_output = weighted[0]
        weighted_confidence = weighted[1]
        
        # calculate iou
        unet_iou = calculate_iou(label,unet_output)
        segnet_iou = calculate_iou(label,segnet_output)
        wcid_iou = calculate_iou(label,wcid_output)

        
        pixelwise_iou = calculate_iou(label,pixelwise_output)
        weighted_iou = calculate_iou(label,weighted_output)

        # save metrics
        iou_unet.append(unet_iou)
        iou_segnet.append(segnet_iou)
        ioU_wcid.append(wcid_iou)
        
        iou_pixelwise_.append(pixelwise_iou)
        iou_weighted_.append(weighted_iou)
    
        
        confidence_unet.append(unet_confidence)
        confidence_segnet.append(segnet_confidence)
        confidence_wcid.append(wcid_confidence)
        confidence_pixelwise.append(pixelwise_confidence)
        confidence_weighted.append(weighted_confidence)
        
        # plot individual metrics
        model_names = ['VGG16-Unet', 'MobileNet-SegNet', 'WCID', ]
        iou_values = [unet_iou, segnet_iou, wcid_iou]
        confidence_values=[unet_confidence, segnet_confidence, wcid_confidence]
        save_path = f"{output_pth}/{img.stem}/metrics_{img.stem}.png"
        plot_and_save_sing_metrics(model_names, iou_values, confidence_values, save_path,f"Scores by Model and Metric for img {img.stem}")
        
        # plot combined metrics
        model_names = ['max pixelweise\n Kombination', 'gewichtete Kombination',]
        iou_values = [pixelwise_iou, weighted_iou]
        confidence_values=[pixelwise_confidence, weighted_confidence]
        save_path = f"{output_pth}/{img.stem}/metrics_eval_{img.stem}.png"
        plot_and_save_sing_metrics(model_names, iou_values, confidence_values, save_path,f"Scores by Evaluation and Metric for img {img.stem}")
        
        cv2.imwrite(f"{output_pth}/{img.stem}/label_{img.stem}.png",label)
        
        # write values to file
        with open(f"{output_pth}/{img.stem}/confidence_{img.stem}.txt","w") as f:
            f.write(f"Unet: {unet_confidence}\n")
            f.write(f"Segnet: {segnet_confidence}\n")
            f.write(f"WCID: {wcid_confidence}\n")
            f.write(f"Pixelwise: {pixelwise_confidence}\n")
            f.write(f"Weighted: {weighted_confidence}\n")
            f.write(f"Unet IoU: {unet_iou}\n")
            f.write(f"Segnet IoU: {segnet_iou}\n")
            f.write(f"WCID IoU: {wcid_iou}\n")
            f.write(f"Pixelwise IoU: {pixelwise_iou}\n")
            f.write(f"Weighted IoU: {weighted_iou}\n")
            
        
    # plot mean metrics of individual models    
    model_names = ['VGG16-Unet', 'MobileNet-SegNet', 'WCID' ]
    iou_values = [iou_unet, iou_segnet, ioU_wcid]
    confidence_values = [confidence_unet, confidence_segnet, confidence_wcid]
    save_path = f"{output_pth}/metrics.png"
    plot_and_save_metrics(model_names, iou_values, confidence_values, save_path,
                          "Durschnittliche Metrikwerte nach Modell")    
    
    # plot mean metrics of n-version
    model_names = ['max pixelweise\n Kombination', 'gewichtete Kombination',]
    iou_values = [iou_pixelwise_, iou_weighted_]
    confidence_values = [confidence_pixelwise, confidence_weighted]
    save_path = f"{output_pth}/metrics_eval.png"

    plot_and_save_metrics(model_names, iou_values, confidence_values, save_path,
                          "Durschnittliche Metrikwerte nach Modell")    
    
    

        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make predictions with N_Version class.')
    parser.add_argument('--img_width', type=int, required=True, help='Width of the image.')
    parser.add_argument('--img_height', type=int, required=True, help='Height of the image.')
    parser.add_argument('--output_pth', type=str, required=True, help='Output path to save the predictions.')
    parser.add_argument('--image_path', type=str, required=True, help='Path of the image to predict.')
    parser.add_argument('--vgg_path', type=str, required=True, help='Path to the VGG model.')
    parser.add_argument('--segnet_path', type=str, required=True, help='Path to the MobileNet model.')
    parser.add_argument('--wcid_path', type=str, required=True, help='Path to the WCID model.')
    parser.add_argument('--th', type=float, required=True, help='Threshold for the predictions.')
    args = parser.parse_args()
    n_version = N_Version(vgg_path=args.vgg_path, mobilenet_psth=args.segnet_path, WCID_path=args.wcid_path,threshold=args.th)

    main( args.img_width, args.img_height, args.output_pth, args.image_path,n_version)

