# train/__main__.py
"""
-------------------------------- DEEPAXON --------------------------------
Main entrypoint for DeepAxon++ training.
Calls command line interface cli_interactive.py

__main__.py
└─> run_interactive() (CLI)

CLI collects:
    - training_dir → verify raw images/masks
    - model_dir
    - model_name
    - epochs
    - test_fraction
    - augmentation choice

CLI initializes DataPipeline (or calls train.train_model()):
    - pipeline verifies raw dataset internally if needed
    - pipeline preprocesses train/val → patches created
    - pipeline exposes total patch counts

CLI calculates recommended batch size using total patches
CLI asks user to confirm/override batch size

train_model():
    - builds model
    - compiles with metrics
    - runs model.fit() using pipeline.get_batches()
    - saves model
    
"""
import os
# suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from train.utils.gpu import setup_gpu_console
setup_gpu_console()

from .utils.cli_interactive import run_interactive

def main():
    run_interactive()

if __name__ == "__main__":
    main()