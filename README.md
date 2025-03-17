## U2NET Model

## Install

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```


## How to train?

**Training Dataset setup**

#### 1. Create a `train_data` folder in the main U2NET directory.

#### 2. Inside the `train_data` folder, create subfolders for the training dataset:
   - `originals`
   - `masks`
##### Note
after creating the folders we will put the training path in `config/config{1, 2, 3 ..}.ini` file. see section 5

#### 3. Example
```python
    train_data/
    ├── originals/
    |    ├── *.jpg
    ├── masks/
    |    ├── *.png

```
#### 4. saving the models
```
    save_model/
    ├── i1/
    |    ├── *.pth
    ├── i2/
    |    ├── *.pth
    ├── i3/
    |    ├── *.pth
    ├── mini/
    |    ├── *.pth

In this pipeline, i1, i2, and i3 will store instance weights. At the end, it moves the top 3 minimal weights to the 'mini' folder. You can configure these paths as desired.
```
##### Note
after creating the folder we will put the save models path in `config/config{1, 2, 3 ..}.ini` file. see section 5

#### 5. Model Training Configuration Guide

```
    The training pipeline is configured through a series of parameters defined in a configuration config/config{1,2,3}.ini file. Key sections and parameters include:

    Model Training
        Epochs: 400
        Batch Size (Train): 16

    Regularization Parameters
        Epsilon: 1e-08
        Weight Decay: 0.001

    Optimizer
        Learning Rate: 0.5
        Betas: (0.9, 0.999)

    Data Loading
        Image Extension: .jpg
        Label Extension: .png

    Model Settings
        Model Type: U2Net
        Debug Mode: False
        Visualize Filters: False

    Paths
        Training, validation, and testing data paths
        Model saving and debug paths

    CUDA
        GPU: 0

    Testing
        Input and Output paths for testing
```
#### 6. u2net file structure:
    
        1) `scheduler.py` you will implement different learning rate scheduler
        2) `data_augmentation.py` you will find all the augmentation implementation in this file
        3) `main.py` you will find the main point of starting and apply the augmentation , initializing the model etc..
        4) In the `config/` folder, you will find the `config{1,2,3}.ini` files, which you will use to set hyperparameters for the training and to run each file on separate GPUs.
        5) In the `runs/` folder you will find the training and validation loss graphs

    
#### 7. start training using following command:
    - `$ CONFIG_FILE=config/config.ini python3 main.py`

## Monitoring

**TensorBoard Monitoring**

1. To monitor the training progress in TensorBoard, run the following command:

    - `$ tensorboard --logdir=runs/ --load_fast true`


## Testing Multiple Models

1. ***Create a folder in the main U2NET directory.***

2. ***Inside the folder, create subfolders for the testing dataset:***

3. ***Example***
```python
    test_data/
    ├── input_images/
    |    ├── *.jpg
    ├── results/
    |    ├── *.png
```
4. ***start testing using following command:*** ` $ CONFIG_FILE=config/config2.ini python3 multi_model_tests.py -l saved_models/u2net/u2net_1.pth saved_models/u2net/u2net_model.pth `


    - -l  `stands for list of weights`

5. ***Minimum Point Evaluation:***
    
    ```
    If you want to test each minimum point during the training, ensure to set the testing flag to 'true' in the config/config{1, 2, 3}.ini file.
    ```


[//]: # (### Data augmentation Testing)

[//]: # ()
[//]: # (To test the data augmentation, run the following command:)

[//]: # ()
[//]: # (- path to dataset should look as follow)

[//]: # (```python)

[//]: # ()
[//]: # (data/)

[//]: # (├── images/)

[//]: # (│   ├── *.jpg)

[//]: # (│   └── *.png)

[//]: # (├── masks/)

[//]: # (│   ├── *.jpg)

[//]: # (│   └── *.png)

[//]: # (├── results/)

[//]: # (│   ├── images/)

[//]: # (│   │   ├── *.jpg  )

[//]: # (│   └── masks/)

[//]: # (│       ├── *.png)

[//]: # (```)

[//]: # (- `$ python3 data_augmentation_test.py /path/to/dataset`)