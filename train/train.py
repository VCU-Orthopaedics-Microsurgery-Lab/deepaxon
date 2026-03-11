# train/train.py
"""
Core DeepAxon training script.
Uses DataPipeline for verified & preprocessed datasets.
Tracks Dice + IoU for both training and validation.
Supports early stopping and LR reduction on validation IoU.
"""

import os
import tensorflow as tf
from rich.console import Console

from .data.pipeline import DataPipeline
from .utils.metrics import dice_coef, iou_coef, combined_loss
from .models.unet import UNET
from .models.unet_plus_plus import UNET_PLUS_PLUS

console = Console()

def train_model(
    training_dir: str,
    model_dir: str,
    model_name: str,
    epochs: int = 100,
    batch_size: int = 8,
    input_shape=(256, 256, 1),
    num_classes: int = 3,
    model_type: str = "unet++",
    test_fraction: float = 0.3,
    augment: bool = True,
    patch_size: int = 256,
):
    """
    Run full training pipeline.

    Parameters:
        training_dir: Path to training dataset (images + masks)
        model_dir: Path to save trained model
        model_name: Name of the saved model
        epochs: Number of training epochs
        batch_size: Batch size
        input_shape: Input image shape
        num_classes: Number of output classes
        model_type: "unet" or "unet++"
        test_fraction: Fraction of dataset for validation
        augment: Apply augmentation during training
        patch_size: Patch size for preprocessing
    """

    # ------------------------------ Prepare Data ---------------------------- #
    console.rule("[bold cyan]PREPARING DATA PIPELINE[/bold cyan]")
    
    pipeline = DataPipeline(
        training_dir,
        test_fraction=test_fraction,
        augment=augment,
        patch_size=patch_size,
    )

    total_train = len(pipeline.train_images)
    total_val = len(pipeline.val_images)
    console.print(f"[green]✅ Found {total_train} training pairs and {total_val} validation pairs[/green]\n")
    
    # ------------------------------ Build Model ----------------------------- #
    console.rule("[bold cyan]BUILDING MODEL[/bold cyan]")
    model_class = UNET_PLUS_PLUS if model_type.lower() == "unet++" else UNET
    model = model_class(input_shape=input_shape, num_classes=num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=combined_loss,
        metrics=[dice_coef, iou_coef]
    )
    model.summary()

    # ------------------------------ Callbacks ------------------------------- #
    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_iou_coef",
        patience=15,
        mode="max",
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_iou_coef",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        mode="max",
        verbose=1
    )

    callbacks = [early_stop_cb, reduce_lr_cb]

    # ------------------------------ Training Loop --------------------------- #
    console.rule("[bold cyan]STARTING TRAINING[/bold cyan]")

    train_gen = pipeline.get_batches(batch_size=batch_size, training=True)
    val_gen = pipeline.get_batches(batch_size=batch_size, training=False)

    steps_per_epoch = max(1, total_train // batch_size)
    validation_steps = max(1, total_val // batch_size)

    history = model.fit(
        train_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    console.rule("[bold green]TRAINING COMPLETE[/bold green]")

    # Save model
    model_save_path = os.path.join(model_dir, f"{model_name}.h5")
    model.save(model_save_path)
    console.print(f"[bold green]Model saved at {model_save_path}[/bold green]")

    return history
