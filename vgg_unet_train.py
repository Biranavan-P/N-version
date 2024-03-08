'''
Autor: Biranavan Parameswaran
'''
from vgg_unet.vgg_unet import VggUnetModel
import argparse
import os
from config.vgg_unet.vgg_config import VGG_Unet_Config
from dataset.dataset import Dataset
import vgg_unet.vgg_unet_predict as vgg_unet_predict

def main():
    model = VggUnetModel()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_height", type=int, default=None, help="Input height")
    parser.add_argument("--input_width", type=int, default=None, help="Input width")
    parser.add_argument("--result_path", type=str, default="results/vgg_unet", help="Result path")
    parser.add_argument("--log_path", type=str, default="logs/vgg_unet", help="Log path")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--epoch", type=int, default=None, help="number of epoch")
    parser.add_argument("--lr",type=float,default=-1,help="learning rate")
    parser.add_argument("--num_classes",type=int,default=3,help="number of classes")
    parser.add_argument("--train_images_path", type=str,
                      
                        help="Train images path")
    parser.add_argument("--train_masks_path", type=str,help="Train masks path")
    parser.add_argument("--val_images_path", type=str, help="Validation images path")
    parser.add_argument("--val_masks_path", type=str, help="Validation masks path")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer")
    args = parser.parse_args()
    input_height = args.input_height
    input_width = args.input_width
    batch_size = args.batch_size
    train_images_path = args.train_images_path
    train_masks_path = args.train_masks_path
    val_images_path = args.val_images_path
    val_masks_path = args.val_masks_path
    epoch = args.epoch
    num_classes = args.num_classes
    
    
    result_path = args.result_path
    log_path = args.log_path
    cfg = VGG_Unet_Config()
    
    
    
    if input_height is not None:
        cfg.input_height = input_height
        assert cfg.input_height == input_height
    if input_width is not None:
        cfg.input_width = input_width
        assert  cfg.input_width == input_width
    if batch_size is not None:
        cfg.batch_size = batch_size
        assert cfg.batch_size == batch_size
    if epoch is not None:
        cfg.epochs = epoch
        assert cfg.epochs == epoch
    if num_classes is not None:
        cfg.n_classes = num_classes
        assert cfg.n_classes == num_classes
    cfg.learning_rate = args.lr
    cfg.optimizer = args.optimizer
    dataset = Dataset(
                    train_img=train_images_path,train_annotation=train_masks_path,
                    valid_img=val_images_path,valid_annotation=val_masks_path
                      )
    dataset.summary()
    
    model.train_model(
                result_path = result_path,
                dataset=dataset,
                log_path = log_path,
                cfg= cfg
                )
   
    
if __name__ == "__main__":
    main()