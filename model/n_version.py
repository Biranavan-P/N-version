'''
Autor: Biranavan Parameswaran
'''
from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.models.segnet import mobilenet_segnet
import os 
from pathlib import Path
import cv2
from wcid_keras.predict import predict_images,predict_image
import tensorflow as tf
import yaml
import numpy as np
from PIL import Image
import argparse

#remove warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)





def compute_weights(confidence):
    # normalize confidences
    confidences = np.array(confidence)
    confidences /= np.sum(confidences)

    # round the normalized confidences to ensure sum equals 1
    confidences = np.round(confidences, decimals=6)
    confidences /= np.sum(confidences)

    return confidences


def filter_probability_by_threshold(probability_map, threshold):
    return np.where(probability_map > threshold, probability_map, 0)


def calculate_iou(input1, input2):
    assert input1.shape == input2.shape, f"Both input images should have the same shape instead of {input1.shape} and {input2.shape}"

    input1 = (input1 == 255).astype(np.uint8)
    input2 = (input2 == 255).astype(np.uint8)

    
    intersection = np.logical_and(input1, input2)

    union = np.logical_or(input1, input2)

    # add a small epsilon to the denominator to avoid division by zero
    iou_score = np.sum(intersection) / (np.sum(union) + 1e-10)

    return iou_score

def calculate_average_confidence( probability_map):
    
    # extract non-zero (railway) probabilities
    foreground_probability = probability_map[probability_map > 0]
    
    # check if there are no railway pixels
    if foreground_probability.size == 0:
        return 0.5
    
    average_probability = np.mean(foreground_probability)

    return average_probability

def calculate_output_confidence(output,iou_1_2,iou_1_3,iou_2_3,threshold):
    output = filter_probability_by_threshold(output, threshold)
    if np.unique(output).size == 1:
        return 0
    confidence = calculate_average_confidence(output)
    mean_iou = (iou_1_2+iou_1_3+iou_2_3)/3
    end_confidence = confidence * mean_iou
    return end_confidence



class N_Version:
    def __init__ (self,vgg_path=None,mobilenet_psth=None,WCID_path=None,threshold=0.6):
        self.vgg_path = vgg_path
        self.mobilenet_psth = mobilenet_psth
        self.WCID_path = WCID_path
        self.threshold = threshold
    
    def predict(self, n_classes, img_width, img_height, output_pth, image_path,save_img=False):
        unet, segnet, wcid = self.get_models(n_classes=n_classes, input_height=img_height, input_width=img_width)
        img_list = Path(image_path)
        if img_list.is_file:
            img_cv = cv2.imread(str(image_path))
            if img_cv is None:
                raise ValueError(f"Image {image_path} could not be read")
            img_name = img_list.stem
            unet_output = unet.predict_segmentation(inp=img_cv, prob_seg_arr=True, resize=True)
            segnet_output = segnet.predict_segmentation(inp=img_cv, prob_seg_arr=True, resize=True)
            wcid_output = predict_image(image_path, wcid, save_pth=None, vis_type="binary",
                                        thd=self.threshold, img_res=(img_width, img_height),
                                        prob_seg_arr=True)
            
            
            # resize outputs to the same size
            unet_output = cv2.resize(unet_output, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
            segnet_output = cv2.resize(segnet_output, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
            wcid_output = cv2.resize(wcid_output, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
            
            
            # make all pixels under threshold background class
            unet_thresholded = filter_probability_by_threshold(unet_output, self.threshold)
            segnet_thresholded = filter_probability_by_threshold(segnet_output, self.threshold)
            wcid_thresholded = filter_probability_by_threshold(wcid_output, self.threshold)
            # malculate confidence scores
            unet_threshold_confidence = calculate_average_confidence(unet_thresholded)
            segnet_threshold_confidence = calculate_average_confidence(segnet_thresholded)
            wcid_threshold_confidence = calculate_average_confidence(wcid_thresholded)
            
            
            
            # calculate iou scores
            unet_segnet_iou = calculate_iou(unet_thresholded, segnet_thresholded)
            unet_wcid_iou = calculate_iou(unet_thresholded, wcid_thresholded)
            segnet_wcid_iou = calculate_iou(segnet_thresholded, wcid_thresholded)
            
            
            
            output = [unet_output, segnet_output, wcid_output]
            confidences = [unet_threshold_confidence, segnet_threshold_confidence, wcid_threshold_confidence]
            # compue weights for weighted combination
            weights = compute_weights(confidences)

            # combine predictions
            pixelwise_output = self.pixel_wise_evaluator(output,self.threshold)
            weighted_output = self.evaluator_weighted_average(output,weights,self.threshold)

            # calculate combined confidence scores
            pixelwise_confidence = calculate_output_confidence(pixelwise_output,unet_segnet_iou,
                                                               unet_wcid_iou,segnet_wcid_iou,self.threshold)
            weighted_confidence = calculate_output_confidence(weighted_output,unet_segnet_iou,unet_wcid_iou,segnet_wcid_iou,self.threshold)
            
            
            # threshold final outputs
            unet_output = np.where(unet_output > self.threshold, 255, 0)
            segnet_output = np.where(segnet_output > self.threshold, 255, 0)
            wcid_output = np.where(wcid_output > self.threshold, 255, 0)
            
            pixelwise_output = np.where(pixelwise_output > self.threshold, 255, 0)
            weighted_output = np.where(weighted_output > self.threshold, 255, 0)


            
            unet_output = unet_output.astype(np.uint8)
            segnet_output = segnet_output.astype(np.uint8)
            wcid_output = wcid_output.astype(np.uint8)
            
            pixelwise_output = pixelwise_output.astype(np.uint8)
            weighted_output = weighted_output.astype(np.uint8)
            
            
            # save images
            if save_img:
                os.makedirs(os.path.join(output_pth, f'{img_name}/individual_output'), exist_ok=True)
                os.makedirs(os.path.join(output_pth, f'{img_name}/evaluator'), exist_ok=True)
                

                cv2.imwrite(os.path.join(output_pth, f'{img_name}/evaluator/pixel_wise_output.png'), pixelwise_output)
                cv2.imwrite(os.path.join(output_pth, f'{img_name}/evaluator/weighted_average_output.png'), weighted_output)
                            
                
                cv2.imwrite(os.path.join(output_pth, f'{img_name}/individual_output/unet_output.png'), unet_output)
                cv2.imwrite(os.path.join(output_pth, f'{img_name}/individual_output/segnet_output.png'), segnet_output)
                cv2.imwrite(os.path.join(output_pth, f'{img_name}/individual_output/wcid_output.png'), wcid_output)
                os.symlink(image_path, os.path.join(output_pth, f'{img_name}/{img_name}.jpg'))
            
            
            return {
                'unet': (unet_output, unet_threshold_confidence),
                "segnet": (segnet_output, segnet_threshold_confidence),
                "wcid": (wcid_output, wcid_threshold_confidence),
                "pixelwise": (pixelwise_output, pixelwise_confidence),
                "weighted": (weighted_output, weighted_confidence),
            }

    def get_models(self,n_classes,input_height,input_width):
        unet = vgg_unet(n_classes=n_classes, input_height=input_height, input_width=input_width)
        unet.load_weights(self.vgg_path)
        
        segnet = mobilenet_segnet(n_classes=n_classes, input_height=input_height, input_width=input_width)
        segnet.load_weights(self.mobilenet_psth)
        
        wcid = tf.keras.models.load_model(self.WCID_path, compile=False)
        
        return unet,segnet, wcid
             
    def pixel_wise_evaluator(self,output:list,threshold):
        # find the maximum probability for each pixel across all models
            output1 = output[0]
            output2 = output[1]
            output3 = output[2]
            max_output = np.maximum(np.maximum(output1, output2), output3)
            return max_output
    
    
    def evaluator_weighted_average(self, outputs:list, weights,threshold):
        assert len(weights) == 3, "Provide a weight for each model"
        assert abs(sum(weights) - 1) < 1e-6, f"Weights should sum up to 1 but they sum up to {sum(weights)}, the weights are {weights}"
        
        # multiply each output by its weight and sum them up
        weighted_output = weights[0]*outputs[0] + weights[1]*outputs[1] + weights[2]*outputs[2]
        
        return weighted_output
   

                 
             

def main(img_width, img_height, output_pth, image_path,model: N_Version):
    model.predict(3, img_width, img_height, output_pth, image_path,True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make predictions with N_Version class.')
    # default values, should not be changed
    parser.add_argument('--img_width', type=int, required=False, help='Width of the image.',default=1920)
    parser.add_argument('--img_height', type=int, required=False, help='Height of the image.',default=1056)
    parser.add_argument('--vgg_path', type=str, required=False, help='Path to the VGG model.',default="vgg_unet/weights/epochs.00058")
    parser.add_argument('--segnet_path', type=str, required=False, help='Path to the MobileNet model.',default="mobilenet_segnet/weights/epochs.00006")
    parser.add_argument('--wcid_path', type=str, required=False, help='Path to the WCID model.',default="model/wcid_keras/weight/checkpoint_076.hdf5")
    parser.add_argument('--th', type=float, required=False, help='Threshold for the predictions.',default=0.6)
    
    
    parser.add_argument('--output_pth', type=str, required=True, help='Output path to save the predictions.')
    parser.add_argument('--image_path', type=str, required=True, help='Path of the image to predict.')
    args = parser.parse_args()
    n_version = N_Version(vgg_path=args.vgg_path, mobilenet_psth=args.segnet_path, WCID_path=args.wcid_path,threshold=args.th)

    main(args.img_width, args.img_height, args.output_pth, args.image_path,n_version)

        
        
            
            
        
    
        
        

    
    
    
    
    