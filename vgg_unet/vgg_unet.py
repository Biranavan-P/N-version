'''
Autor: Biranavan Parameswaran
'''
import os
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint,EarlyStopping
from model.keras_segmentation.models.unet import vgg_unet
from config.vgg_unet.vgg_config import VGG_Unet_Config
from dataset.dataset import Dataset
from keras import backend as K
from model.keras_segmentation.train import train
from .utils import get_metrics_multi_class,get_callbacks 
from .vgg_unet_predict import main as predict_main


def calculate_steps(dir_path, batch_size):
    # calculate steps per epoch
    count = 0

    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path,path)):
            count += 1
    
    steps = int(count / batch_size)

    return steps


    
    
class VggUnetModel():
    
    def train_model(self, result_path:str,
                dataset:Dataset,
             log_path:str,
             cfg:VGG_Unet_Config
             ):   
        
        
        n_classes = cfg.n_classes
        # safety mechanism to prevent the usage of cpu
      
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) == 0:
            raise RuntimeError("GPU not found",gpus)


        
        # calculate steps per epoch
     
        steps_per_epoch = calculate_steps(dataset.train_img, cfg.batch_size)
        val_steps_per_epoch = calculate_steps(dataset.valid_img, cfg.batch_size)
        
        

       # define model
    

        
        model ="vgg_unet"
        opt_name = cfg.optimizer.lower()
        if opt_name == "adam":
                optimizer = tf.keras.optimizers.Adam()
        elif opt_name == "rmsprop":
                optimizer = tf.keras.optimizers.RMSprop()
        elif opt_name == "sgd":
                optimizer= tf.keras.optimizers.SGD()
        else:
            raise ValueError("Invalid optimizer")
       
        if cfg.learning_rate != -1:
            optimizer.learning_rate = cfg.learning_rate
        cfg.print_config()
        lr = np.float32(optimizer.learning_rate.numpy())
        print("------------------"*20)
        print(f"optimizer: {optimizer.name}")
        print(f"learning_rate: {lr}")
        
        log_path+=f"-optimizer_{optimizer.name}-lr_"+str(lr)  
        result_path+=f"-optimizer_{optimizer.name}-lr_"+str(lr)
        # add momentum and nesterov if optimizer allows it
        if "momentum" in optimizer.__dict__: 
            optimizer.momentum = 0.9
            log_path+="-momentum_"+str(optimizer.momentum)
            result_path+="-momentum_"+str(optimizer.momentum)
            
        if "nesterov" in optimizer.__dict__:
            optimizer.nesterov = True
            log_path+="-nesterov_"+str(optimizer.nesterov)
            result_path+="-nesterov_"+str(optimizer.nesterov)
        
            
            
        
        print("------------------"*20)
        print(f"model: {model}")
        print("result_path: ",result_path)
        # train model
        train(
            model=model,
            input_height=cfg.input_height,
            input_width=cfg.input_width,
            n_classes=n_classes,
            train_images       =  dataset.train_img,
            train_annotations  =  dataset.train_annotation,
            load_weights       = None,
            checkpoints_path   = result_path,
            validate           = True,
            val_images         =  dataset.valid_img,
            val_annotations    =  dataset.valid_annotation,
            batch_size = cfg.batch_size,
            val_batch_size     =  cfg.batch_size,
            steps_per_epoch    = steps_per_epoch,
            val_steps_per_epoch= val_steps_per_epoch,
            epochs             = cfg.epochs,
            metrics=get_metrics_multi_class(n_classes),
            callbacks=[get_callbacks(log_path, result_path)],
            optimizer_name=optimizer,
        )
       

        
        
        
    
    