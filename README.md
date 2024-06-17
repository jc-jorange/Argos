# Argos
*"Assorted Recognition General Objects Surveillance System"*

A Multiple Camera Multiple Object Tracking System.

## Setup
### Requirements
- Python >= 3.9
- Pytorch >= 1.13.1

### Installation

1. Clone this repository.
    ```
    git clone 
    ```

2. Setup environment.  
    Here using Anaconda as example:
    ```
    conda env create -n Argos
    conda activate Argos
    ```

    Make sure you install Pytorch in your environment

    ```
    pip install cython
    pip install -r requirements.txt
    mim install mmcv  # Use mmcv for DCNv2
    ```

## Project Structure
The whole project is based on a multiprocessing pipeline system, which can be easily configured in *YAML* format.
Project structure as following:

```
src
├─── dataset
├─── trainer
├─── model
│   ├─── cfg
│   └─── networks
├─── multiprocess_pipeline
│   ├─── cfg
│   ├─── process
│   ├─── shared_structure
│   └─── workers
└─── opts
```

  - **Dataset**

    All dataset related is in `dataset` directory. Substructure as following:

    ```
    dataset
    ├─── cfg
    ├─── convert
    ├─── data_path
    └─── __init__
    ```
    
    `cfg`: Store dataset config file by *JSON*.
    
    `convert`: Contain all utils to convert dataset format to *MOT Challenge*.

    `data_path`: Store image file paths by specialized file type.

    `__init__`: All dataset class defined in this file.


  - **Trainer**

    `trainer` defined basic trainer class for network training. Substructure as following:

    ```
    trainer
    ├─── utils
    └─── __init__
    ```
    
    `utils`: Util functions, for example: initial check.

    `__init__`: Defined all trainers.


  - **Model**

    `model` contain all network related content. 

    Current neural network in this project is also using module pipeline system as following:

    `Backbone --> Neck --> Head`

    or

    `(Backbone + Neck) --> Head`.
    
    We divided networks into `backbone`, `neck`, `backbone_with_neck` and `head` 4 categories module parts.
    A simple swappable module system is achieved. 
    
    Here is a example configure file
    `DLA+GhostPAN_mot_8class.yml`
    :
    ```
    _description: ''
    backbone:
      _description: ''
      cfg_name: default
      model_name: DLA
    backbone_with_neck:
      _description: ''
      cfg_name: ''
      model_name: ''
    head:
      _description: ''
      cfg_name: default
      model_name: FairMOT
    max_classes_num: 8
    max_objects_num: 500
    neck:
      _description: ''
      cfg_name: default
      model_name: Ghost_PAN
    ```

    Substructure of this `model` directory as following:

    ```
    model
    ├─── cfg
    ├─── networks
    │    ├─── backbone
    │    │    ├─── DLA
    │    │    ├─── ParNet
    │    │    ├─── ShuffleNetV2
    │    │    ├─── ...
    │    ├─── backbone_with_neck
    │    │    ├─── DLA_DCN
    │    │    ├─── ParNet
    │    │    ├─── ResNet_DCN
    │    │    ├─── ...
    │    ├─── head
    │    │    └─── FairMOT
    │    ├─── loss
    │    ├─── module
    │    ├─── neck
    │    │    ├─── DLA_Fusion
    │    │    ├─── FPN
    │    │    ├─── Ghost_PAN
    │    │    ├─── ...
    │    └─── model_part_config.py
    ├─── utils
    └─── model_config.py
    ```
    
    `cfg`: store model config files.

    
  - **Multiprocess Pipeline**
  - **Opts**

## Usage
- Tracking:
    ```
    python track.py --exp_id Test --pipeline_cfg ./src/multiprocess_pipeline/cfg/TestFunc.yml
    ```
  here:  
  `--exp_id` as experiment name for example here I use `Test`;  
  `--pipeline_cfg` as pipeline config file path.  


  More arguments details please see in `./src/opts/`. 