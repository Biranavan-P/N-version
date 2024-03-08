'''
Autor: Biranavan Parameswaran
'''
import yaml
import os


    
    
class VGG_Unet_Config:
    
    def __init__(self,config_path:str =os.getcwd()+"/params_train.yaml"):
        print("config_path",config_path)
        self.input_height = -1
        self.input_width= -1
        self.batch_size= -1
        self.epochs= -1
        self.learning_rate = 0.0001
        self.n_classes = -1
        self.optimizer = ""
        
        self.load_config(config_path)
    
    def load_config(self,config_path):
        config = yaml.safe_load(open(config_path))["vgg_unet"]
        
        # check if config file is complete
        file_params = list(config.keys())
        class_params = list(self.__dict__.keys())
        if len(file_params) != len(class_params):
            raise ValueError(f"config file is not complete. Missing parameters: {set(class_params) - set(file_params)}")
            
            
        
        
        # set all hyperparameters attributes from config file
        for key in config:
            setattr(self,key,config[key])
        
        
    def print_config(self):
        print("------------------"*20)
        print("VGG_Unet_Config!")
        for key in self.__dict__.keys():
            
            # if lr is set to -1, then the default value of optimizer is used
            if key =="learning_rate" and getattr(self,key) == -1:
                print("learrning rate is set to default value of optimizer")
            else:
                print(f"{key}: {getattr(self,key)}")
        
        
        