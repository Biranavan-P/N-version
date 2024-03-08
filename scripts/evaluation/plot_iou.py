'''
Autor: Biranavan Parameswaran
'''
import pandas as pd
import matplotlib.pyplot as plt
import json


json_file = './wcid_merged.json'  
model_name ="WCID"
IoU="epoch_IoU rail"
IoU_title="IoU rail"

with open(json_file) as file:
    data = json.load(file)


# plotting mean IoU excluding background for all models in one plot
fig, axs = plt.subplots(figsize=(10, 7))

for model in sorted(data.keys()):
    
    mean_iou_excl_background = pd.Series(data[model]['val'].get(IoU, {})).astype(float)
    
    mean_iou_excl_background.index = mean_iou_excl_background.index.astype(int)
    
    axs.plot(mean_iou_excl_background, label=model)

axs.set_title(f'{model_name} {IoU_title}')
axs.set_xlabel('Epoche')
axs.set_ylabel(IoU_title)
axs.legend()

plt.tight_layout()
plt.savefig(f'{model_name}_mean_iou_excl_background.png')


fig, axs = plt.subplots(figsize=(10, 7))

for model in sorted(data.keys()):
    mean_iou_excl_background = pd.Series(data[model]['val'].get(IoU, {})).astype(float)
    
    mean_iou_excl_background.index = mean_iou_excl_background.index.astype(int)
    
    # select IoUs up to the best epoch (40 before the last)
    best_epoch = max(mean_iou_excl_background.index) - 40
    mean_iou_excl_background = mean_iou_excl_background[mean_iou_excl_background.index <= best_epoch]

    # plot the mean IoU excluding background until the best epoch
    axs.plot(mean_iou_excl_background, label=model)

axs.set_title(f'{model_name} {IoU_title} bis zur besten Epoche')
axs.set_xlabel('Epoche')
axs.set_ylabel(IoU_title)
axs.legend()

plt.tight_layout()
plt.savefig(f'{model_name}_mean_iou_excl_background_best_epoch.png')
