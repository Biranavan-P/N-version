'''
Autor: Biranavan Parameswaran
'''
from tensorflow.python.summary.summary_iterator import summary_iterator
import json
from pathlib import Path
from tensorflow.core.util import event_pb2
import tensorflow as tf
path = "./model/result/eval/results_height_1056-width_1920-batch_size_16-epoch_400-loss_bce-optimizer_adam-eary_stopping-patience-40-lr_schedular/"
sub_folder = [f for f in Path(path).iterdir() if f.is_dir()]
tags = []

metrics = ["loss", "epoch_loss","one hot mean IOU",
                          "one hot mean IOU excluding Background",
                          "background one hot IOU",
                          "rail-track one hot IOU",
                          "rail-raised one hot IOU",
                          "mean",
                          "rail",
                          "background",
                          ]
# extract all the data from the tensorboard logs
for sub in sub_folder:
    name = sub.name
    files = [f for f in sub.rglob("*.out.*") if f.is_file()]
    evaluation = {
        "name": name,
    }
    
    for file in files:
        split = "train" if "train" in str(file) else "val"
        evaluation[split] = {}
        serialized_examples = tf.data.TFRecordDataset(file)
        for serialized_example in serialized_examples:
            event = event_pb2.Event.FromString(serialized_example.numpy())
            
            for value in event.summary.value:
                tags.append(value.tag)
                t = tf.make_ndarray(value.tensor)
                
                if any(substring in value.tag for substring in metrics) and "vs_iterations" not in value.tag:
                    evaluation[split][value.tag] = evaluation[split].get(value.tag, {}) | {event.step: t.tolist()}
    json.dump(evaluation, open(f"{name}.json", "w"),indent=4)
