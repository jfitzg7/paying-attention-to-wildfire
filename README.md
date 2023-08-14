# cs535-term-project - Wildfire Spread Prediction

The dataset we used in this project can be found here: [https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread)

## File and folder descriptions

* trainModel.py: This file contains our script for training the models using PyTorch DDP.
* datasets.py: Implementations of PyTorch's `Dataset` class that can handle the widlfire data
* old_datasets.py: Older implementations of PyTorch's `Dataset` class that we used early on.
* models.py: Models that we used early on for testing
* milesial_unet_model.py: The first UNet model that we experimented with. Some minor changes have been made to it. The original can be found here [https://github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
* leejunhyun_unet_models.py: The UNet models that we ended up using in our final experiments. No changes were made to the original code, which can be found here [https://github.com/LeeJunHyun/Image_Segmentation](https://github.com/LeeJunHyun/Image_Segmentation)
* pickle_wildfire_datasets.py: Takes random 32x32 crops of the original data and converts it into numpy arrays and pickles them. These are then used by the old `Dataset` implementations.
* pickle_full_wildfire_datasets.py: Converts the original 64x64 data into numpy arrays and pickles them. These are then used by the current `Dataset` implementations.
* runFinal.py: Quality of life script for spinning up multiple nodes to run the trainModel.py script
* testModels.ipynb: A notebook template for testing the models after training them.
* notebooks: This folder contains jupyter notebooks that describe the experiments that we ran.
