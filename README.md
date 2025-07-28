# DeepAxon
Command-line interface automated axon-myelin segmentation and morphometric analysis.

## Installation Guide
1. Install [Python](https://www.python.org/) version 3.8
2. Install [Miniconda](https://docs.anaconda.com/free/miniconda/)
3. Install [git](https://git-scm.com/)
4. Open Miniconda terminal window
5. type the following one line at a time:
````
git clone https://github.com/kushsavsani/deepaxon
cd deepaxon
conda create -n "da_venv" python=3.8
conda activate da_venv
pip install -r requirements.txt
````

## Train a Model
1. Navigate to the deepaxon folder in Miniconda
2. Type `python train`

## Segment Images
1. Navigate to the deepaxon folder in Miniconda
2. Type `python segment`

**The input for segmentation is a folder that contains ONLY images**

## Get Image Morphometrics
1. Navigate to the deepaxon folder in Miniconda
2. Type `python morphometrics`

**The input for morphometric analysis is a folder that contains ONLY segmented images**

*MORE INFORMATION TO COME THE README SOON*