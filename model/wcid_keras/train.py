#!/usr/bin/env python
'''
Autor: Florian Hofstetter
Quelle: https://github.com/FloHofstetter/wcid-keras
Letzter Zugriff: 01.08.2023
Diese Datei wurde durch Biranavan Parameswaran bearbeitet.
'''
from dataset import RailDataset
from model import get_model
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import pathlib
import datetime
def scheduler(epoch, lr):
   if epoch < 10:
     return lr
   else:
     return lr * tf.math.exp(-0.1)

def train(
    trn_img,
    val_img,
    trn_msk,
    val_msk,
    aug_prm,
    sve_pth="output/",
    train_res=(320, 160),
    file_ext="jpg",
    lr=0.0001,
    bs=1,
    epochs=1,
    load_echpoint_pth=None,
    bg_dir_pth = None,
    loss_fn="bce",
    optimizer="adam",
):
    """
    Tran the model, save model and weights.
    Training can continue at former checkpoint.

    :param trn_img: Path to training image folder.
    :param val_img: Path to validation image folder.
    :param trn_msk: Path to training masks.
    :param val_msk: Path to validation masks.
    :param aug_prm: Dict of augmentation parameters.
    :param sve_pth: Path to save model and other artifacts.
    :param train_res: Resolution to train the network. (width, height)
    :param file_ext: Extension for the image/mask data.
    :param lr: Learning rate for the adam solver.
    :param bs: Batch size for training and validation.
    :param epochs: Training epochs to iterate over dataset.
    :param load_echpoint_pth: Path to load checkpoint.
    :param bg_dir_pth: Directory to images for background augmentation.
    :return: None.
    """
    # Get time for save path
    iso_time = datetime.datetime.now().isoformat(timespec="minutes")

    # Root path to save current training
    sve_pth = pathlib.Path(sve_pth)
    sve_pth = sve_pth.joinpath(f"{iso_time}")
    # Path to metrics plot
    history_pth = sve_pth.joinpath(sve_pth, "history/")
    history_pth.mkdir(parents=True, exist_ok=True)
    # Path to epoch checkpoints
    checkpoint_pth = sve_pth.joinpath(sve_pth, f"checkpoints/")
    checkpoint_pth.mkdir(parents=True, exist_ok=True)
    # Path to last model incl. architecture
    last_model_pth = checkpoint_pth.joinpath("end_model.h5")
    # Path to tensorboard log
    tb_log_pth = sve_pth
    tb_log_pth.mkdir(parents=True, exist_ok=True)
    # Path to training csv history
    csv_history_pth = sve_pth.joinpath("history.csv")
    # Path to training plot history
    plt_history_pth = sve_pth.joinpath("history.svg")

    # Paths to store metrics and checkpoints
    history_pth2 = history_pth.joinpath("history2.svg")
    history_pth3 = history_pth.joinpath("history3.csv")
    history_pth = history_pth.joinpath("history.svg")
    checkpoint_pth = pathlib.PurePath(
        checkpoint_pth, "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    )

    # Data generator for training
    trn_gen = RailDataset(
        trn_img,
        trn_msk,
        train_res,
        file_ext,
        msk_ftype="png",
        batch_size=bs,
        tfs_prb=aug_prm,
        bg_dir_pth=bg_dir_pth
    )
    # Data generator for validation
    val_gen = RailDataset(
        val_img,
        val_msk,
        train_res,
        file_ext,
        msk_ftype="png",
        batch_size=bs,
        transforms=False,
    )

    model = get_model(input_shape=train_res)

    # Load former checkpoint if available.
    if load_echpoint_pth is not None:
        load_echpoint_pth = pathlib.Path(load_echpoint_pth)
        model.load_weights(load_echpoint_pth)
    metrics = ["accuracy",
               tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5,name="mean IoU"),
               tf.keras.metrics.BinaryIoU(target_class_ids=[0], threshold=0.5,name="IoU background"),
               tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5,name="IoU rail"),
               
               ]
    # Compile model
    # Choose loss function
    print(f"Using loss function: {loss_fn}")
    if loss_fn == "mse":
        loss = tf.keras.losses.MeanSquaredError()
    elif loss_fn == "mae":
        loss = tf.keras.losses.MeanAbsoluteError()
    elif loss_fn == "bce":
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    # Lower lerning rate every 5th epoch.
    # One step means model optimized to one mini batch aka one iteration.
    decay_steps = int(5 * (len(trn_gen)/bs))
    print(loss.name)

   
    if optimizer == "adam":
        
        opt = tf.keras.optimizers.Adam()
        
        
    elif optimizer == "sgd":
        opt = tf.keras.optimizers.SGD(momentum=0.9, nesterov=True)
    elif optimizer =="rmsprop":
        opt = tf.keras.optimizers.RMSprop(momentum=0.9)
    else:
        msg = f"Unknown optimizer setting: {opt}"
        raise ValueError(msg)        
    if lr !=-1:
        opt.lr = lr
    model.compile(
        loss=loss, optimizer=opt, metrics=metrics,
        
        )

    # Training callbacks
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_pth, save_weights_only=False, verbose=1
    )
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tb_log_pth ,
        histogram_freq=1, 
        write_graph=True,
        write_images=True, write_steps_per_second=True)
    es_callback =tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=40,verbose=1)
    #ls_callback = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose=1)

    
    callbacks = [cp_callback, tb_callback,es_callback,
                 #ls_callback
                 ]
    model.summary()
    for k in (model.__dict__):
        print(k, model.__dict__[k])

    # Train model
    history = model.fit_generator(
        trn_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        workers=12,
        use_multiprocessing=True,
        max_queue_size=100,
    )

    # Save Last model
    model.save(last_model_pth)

    # Save history
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(csv_history_pth)

    # Save history plot
    hist_df.plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.xlabel("epoch in 1")
    plt.ylabel("metric in 1")
    plt.savefig(plt_history_pth)
    hist_df[["val_loss", "acc_loss"]].plot(figsize=(8, 5), logy=True)
    plt.grid(True)
    plt.savefig(history_pth2)


def parse_args(parser: argparse.ArgumentParser):
    """
    Parse CLI arguments to control the training.

    :param parser: Argument parser Object.
    :return: CLI Arguments object.
    """
    parser.add_argument(
        "train_images",
        type=pathlib.Path,
        help="Path to the folder containing RGB training images.",
    )
    parser.add_argument(
        "train_masks",
        type=pathlib.Path,
        help="Path to the folder containing training masks.",
    )
    parser.add_argument(
        "val_images",
        type=pathlib.Path,
        help="Path to the folder containing RGB validation images.",
    )
    parser.add_argument(
        "val_masks",
        type=pathlib.Path,
        help="Path to the folder containing validation masks.",
    )
    parser.add_argument(
        "extension",
        type=str,
        help="Name of the file extension. For example: '-e jpg''.",
    )
    parser.add_argument(
        "background_pth",
        type=pathlib.Path,
        help=(
            "Path to direcotry contatining images used for background switch"
            " augmentation. For example: '/path_to/random_images/'."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        help=(
            "Path to the folder where to store model path and "
            + "other training artifacts."
        ),
        default="output/",
        required=False,
    )
    parser.add_argument(
        "-ep",
        "--epochs",
        type=int,
        help="Training epochs.",
        default=10,
        required=False,
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        help="Learning rate for the adam solver.",
        default=0.0001,
        required=False,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        help="Batch size of training and validation.",
        default=1,
        required=False,
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        help="Select the GPU id to train on.",
        default=0,
        required=False,
    )
    parser.add_argument(
        "--height",
        type=int,
        help="Height of the neural network input layer.",
        default=160,
        required=False,
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        help="Width of the neural network input layer.",
        default=320,
        required=False,
    )
    parser.add_argument(
        "--horizontal_flip",
        type=float,
        help="Probability of flipping image in training set. Default is 0.5.",
        default=0.5,
        required=False,
    )
    parser.add_argument(
        "--brightness_contrast",
        type=float,
        help="Probability of applying random brightness contrast on image in training set. Default is 0.2.",
        default=0.2,
        required=False,
    )
    parser.add_argument(
        "--rotation",
        type=float,
        help="Probability of applying random rotation on image in training set. Default is 0.9.",
        default=0.9,
        required=False,
    )
    parser.add_argument(
        "--motion_blur",
        type=float,
        help="Probability of applying motion blur on image in training set. Default is 0.1.",
        default=0.1,
        required=False,
    )
    parser.add_argument(
        "--background_swap",
        type=float,
        help="Probability of applying background swap on image in training set. Default is 0.9.",
        default=0.9,
        required=False,
    )
    parser.add_argument(
    "--optimizer",
    type=str,
    choices=["adam", "sgd", "rmsprop"],
    help="Optimizer to use for training.",
    default="adam",
    )
    parser.add_argument(
    "--resume_pth", 
    help="Path to the model to resume training from.",
    type=pathlib.Path,
    required=False,
    )
    
    return parser.parse_args()


def main():
    """
    Entry point for training.
    """
    # Parse arguments from cli
    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    # Path parameters
    trn_img = args.train_images
    val_img = args.val_images
    trn_msk = args.train_masks
    val_msk = args.val_masks
    sve_pth = args.output
    bg_pth = args.background_pth
    optimizer= args.optimizer

    # Hardware parameters
    # gpu = args.gpu
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # physical_devices = tf.config.list_physical_devices("GPU")
    # tf.config.experimental.set_memory_growth(physical_devices[gpu], True)

    # Training parameters
    train_res = (args.width, args.height)  # Width height
    lr = args.learning_rate
    bs = args.batch_size
    epochs = args.epochs

    # Augmentation parameters
    aug_prm = {
        "HorizontalFlip": args.horizontal_flip,
        "RandomBrightnessContrast": args.brightness_contrast,
        "Rotate": args.rotation,
        "MotionBlur": args.motion_blur,
        "BackgroundSwap": args.background_swap,
    }
    print(aug_prm)
    train(
        trn_img,
        val_img,
        trn_msk,
        val_msk,
        aug_prm,
        sve_pth=sve_pth,
        epochs=epochs,
        train_res=train_res,
        lr=lr,
        bs=bs,
        bg_dir_pth=bg_pth,
        optimizer=optimizer,        

    )


if __name__ == "__main__":
    main()
