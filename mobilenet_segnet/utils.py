'''
Autor: Biranavan Parameswaran
'''
import os
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint,EarlyStopping,LearningRateScheduler
ONE_HOT_MEAN_IOU = "one hot mean IOU"
ONE_HOT_MEAN_IOU_EXCLUDING_BACKGROUND = "one hot mean IOU excluding Background"
BACKGROUND_ONE_HOT_IOU = "background one hot IOU"
RAIL_TRACK_ONE_HOT_IOU = "rail-track one hot IOU"
RAIL_RAISED_ONE_HOT_IOU = "rail-raised one hot IOU"
def scheduler(epoch, lr):
    # lr schedular from Keras (https://keras.io/api/callbacks/learning_rate_scheduler/). Last accessed: 05.08.2023
   if epoch < 10:
     return lr
   else:
     return lr * tf.math.exp(-0.1)

def get_metrics_multi_class(n_classes=3):
    return[
        tf.keras.metrics.OneHotMeanIoU(num_classes=n_classes,name=ONE_HOT_MEAN_IOU),
        tf.keras.metrics.OneHotMeanIoU(num_classes=n_classes,name=ONE_HOT_MEAN_IOU_EXCLUDING_BACKGROUND,ignore_class=[0]),
        tf.keras.metrics.OneHotIoU(num_classes=3, target_class_ids=[0], name=BACKGROUND_ONE_HOT_IOU),
        tf.keras.metrics.OneHotIoU(num_classes=3, target_class_ids=[1], name=RAIL_TRACK_ONE_HOT_IOU),
        tf.keras.metrics.OneHotIoU(num_classes=3, target_class_ids=[2], name=RAIL_RAISED_ONE_HOT_IOU)
        
        
    ]

                                                                                                   
    
def get_callbacks(log_path, result_path):
    return  [
            TensorBoard(log_dir=log_path, histogram_freq=1, write_graph=True,
                                         write_images=True),
            
            ModelCheckpoint(
                filepath=os.path.join(result_path,"epochs") + ".{epoch:05d}",
                save_weights_only=True,
                verbose=True
            ),
            EarlyStopping(monitor="val_loss",patience=40,verbose=1),
            #LearningRateScheduler(scheduler,verbose=1),


            
        ]