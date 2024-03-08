'''
Autor: Biranavan Parameswaran
'''
import pandas as pd
import matplotlib.pyplot as plt
import json

json_file = './wcid_merged.json'  
model_name = "WCID"  
loss_name = "epoch_loss" 

# remove outliers if necessary (vgg16-unet has some outliers)
remove_outliers = False  

with open(json_file) as f:
    data = json.load(f)

if remove_outliers:
    val_loss_model3 = pd.Series(data['Model 3']['val'][loss_name]).astype(float)
    val_loss_model3.index = val_loss_model3.index.astype(int)

    # identify outliers as values over 100
    outliers = val_loss_model3[val_loss_model3 > 100]

    val_loss_model3 = val_loss_model3.drop(outliers.index)

    data['Model 3']['val'][loss_name] = val_loss_model3.to_dict()

fig, axs = plt.subplots(1, 2, figsize=(20, 7))

for model in sorted(data.keys()):
    val_loss = pd.Series(data[model]['val'][loss_name]).astype(float)
    train_loss = pd.Series(data[model]['train'][loss_name]).astype(float)
    
    val_loss.index = val_loss.index.astype(int)
    train_loss.index = train_loss.index.astype(int)
    
    axs[0].plot(val_loss, label=model)
    
    axs[1].plot(train_loss, label=model)

axs[0].set_title(f'{model_name} Validation Loss')
axs[0].set_xlabel('Epoche')
axs[0].set_ylabel('Validation Loss')
axs[0].legend()

axs[1].set_title(f'{model_name} Training Loss')
axs[1].set_xlabel('Epoche')
axs[1].set_ylabel('Training Loss')
axs[1].legend()

fig.tight_layout()
plt.savefig(f'{model_name}_validation_and_training_loss_side_by_side.png')
