# train/__init__.py
"""
Contains modules for data loading, preprocessing, augmentation,
model definitions, training logic, and CLI utilities.

Package marker; can also define package-level variables if needed

DeepAxon training package structure
train/
├── __init__.py
├── __main__.py
├── train.py
│
├── data/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── augment.py
│   └── pipeline.py
│
└── utils/
│   ├── __init__.py
│   ├── console_utils.py
│   ├── cli_interactive.py
│   ├── metrics.py
│   ├── logger.py
│   ├── gpu.py
│   └── helpers.py
│   
└── models/
    ├── __init__.py
    ├── unet.py (base architechture)
    └── unet_plus_plus.py (default)
    
"""