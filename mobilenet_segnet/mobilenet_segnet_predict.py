'''
Autor: Biranavan Parameswaran
'''
from model.keras_segmentation.models.segnet import mobilenet_segnet
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
import os
def parse_args(parser: argparse.ArgumentParser):
    parser.add_argument(       
        "--result_path",
        type=str,
        help="Path to folder, where the predicted masks should be saved.",
    )
    parser.add_argument(
        "--img_width",
        type=str,
        help="Width of the input images.",
    )
    parser.add_argument(
        "--img_height",
        type=str,
        help="Height of the input images.",
    
    )
    parser.add_argument(    
        "--weights_path",
        type=str,
        help="Path to trained model Folder.",
    )
    parser.add_argument(      
        "--testImg_path",
        type=str,
        help="Path to test images.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        help="Number of classes.",
    )

    return parser.parse_args()

def predict(model, imgs, result_path):
    for img in tqdm(imgs):
        
        out_pth = os.path.join(result_path, img.name)
        print(out_pth)
        img_cv = cv2.imread(str(img))
        out = model.predict_segmentation(inp=img_cv, out_fname=out_pth)

def main():
    parser = argparse.ArgumentParser()
    args = parse_args(parser)
        
    input_height = int(args.img_height)
    input_width = int(args.img_width)
    resul_path = args.result_path
    weights_path = args.weights_path
    testImg_path = args.testImg_path
    num_classes = args.num_classes
    print(input_height, input_width)
    print("predicting images")

    segnet = mobilenet_segnet(n_classes=num_classes, input_height=input_height, input_width=input_width)
    segnet.load_weights(weights_path)

    img_list = Path(testImg_path).iterdir()
    img_list = list(img_list)[:15]
    Path(resul_path).mkdir(parents=True, exist_ok=True)


    
    
    predict(segnet, img_list, resul_path)
    
    
    
if __name__ == "__main__":
    main()
