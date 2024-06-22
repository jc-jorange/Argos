# Argos
*"Assorted Recognition General Objects Surveillance System"*

A Multiple Camera Multiple Object Tracking System.


## Setup
- ### Requirements
  - Python >= 3.9
  - Pytorch >= 1.13.1

- ### Installation

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

## Usage
- Tracking:
    ```
    python track.py --exp_id Test --pipeline_cfg ./src/multiprocess_pipeline/cfg/TestFunc.yml
    ```
    here:  
    `--exp_id` as experiment name for example here I use `Test`;  
    `--pipeline_cfg` as pipeline config file path.  
    
    More arguments details please see in `./src/opts/track.py`. 
  
## Training

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
    └─── dataset.py
    ```
    
    - `cfg`: Store dataset config file by *JSON*.
    - `convert`: Contain all utils to convert dataset format to *MOT Challenge*.
    - `data_path`: Store image file paths by specialized file type.
    - `dataset.py`: All dataset class defined in this file.


  - **Trainer**

    `trainer` defined basic trainer class for network training. Substructure as following:
    ```
    trainer
    ├─── utils
    └─── trainer.py
    ```
    
    - `utils`: util functions, for example: initial check.
    - `trainer.py`: defined all trainers.


  - **Model**

    `model` contain all network related content. 
    We use a simple swappable module system to develop tracking networks. 

    The whole model can be easily defined by a model configure file in `model/cfg`. 
    And each part network also can be defined by its own part configure file in `model/networks/[part category]/[part name]/cfg`.
    
    Substructure as following:
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
    ├─── model_config.py
    ├─── utils
    └─── base_model.py
    ```
    - `cfg`: store overall model config files.
    - `network`: contain all model parts.
    - `model_config.py`: defined model configure structure.
    - `utils`: util functions.
    - `base_model.py`: define the base class of model.
    - Here is an example model configure file `DLA+GhostPAN_mot_8class.yml`:
      ``` yaml
      _description: ''
      backbone:
        _description: ''
        cfg_name: default
        model_name: DLA
      backbone_with_neck:
        _description: ''
        cfg_name: ''
        model_name: ''
      neck:
        _description: ''
        cfg_name: default
        model_name: Ghost_PAN
      head:
        _description: ''
        cfg_name: default
        model_name: FairMOT
      max_classes_num: 8
      max_objects_num: 500
      ```
      - `_description`: human-readable additional information.
      - `model_name`: model part name.
      - `cfg_name`: the part configure file name used for this model part.
      - `max_classes_num`: max tracking object categories in this model.
      - `max_objects_num`: max tracking object amounts for each category in this model. 
  
    An example part network `FairMOT` substructure as following:
    ```
    ...
    ├─── head
    │    ├─── FairMOT
    │    │    ├─── cfg
    │    │    │    └─── default.yml
    │    │    └─── FairMOT.py
    ...  ... 
    │    ├─── _masterclass.py
    │    └─── __init__.py
    ...   
    ```
    - `cfg`: store model configure file. Configure contents are this part model initial structure key arguments. In this example, we have one configure `default.yml`.
    - `[model name].py`: model python file. In this example, we use `FairMOT.py` model.
    - `_masterclass.py`: define the base model of this part.
    - `__init__.py`: all usable networks in this part should be registered in this file to use.

  - **Multiprocess Pipeline**
    
    `multiprocess_pipeline` contain all multiprocess pipeline related contents. In this project, we use a multiprocess based configurable pipeline system for tracking implement. 
    System configure files stored in `multiprocess_pipeline/cfg`.

    Substructure as following:
    ```
    multiprocess_pipeline
    ├─── cfg
    ├─── process
    │   ├─── producer
    │   │   ├─── ...
    │   │   └─── _masterclass.py
    │   ├─── consumer
    │   │   ├─── ...
    │   │   └─── _masterclass.py
    │   ├─── post
    │   │   ├─── ...
    │   │   └─── _masterclass.py
    │   └─── _masterclass.py
    ├─── workers
    │   ├─── image_loader
    │   ├─── tracker
    │   ├─── ...
    │   └─── postprocess
    ├─── shared_structure
    └─── utils
    ```
    
    - `cfg`: store overall implement multiprocess pipeline config files.
    - `process`: contain all -*producer*, *consumer*, *post*- 3 categories process files. The base class define is in `_masterclass.py`
    - `workers`: all function workers files. Some process can have different workers for different methods with the same usage. 
For example, process `ImgaLoader` can have worker `loader_address` load image from an IP address or have worker `loader_video` load image from a video file.
    - `shared_structure`: define all shared memory structure like `data_hub` and `output_port`
    - `utils`: util functions.
    - An example structure configure `TestFunc.yml` as following:
      ``` yaml
      FuncTest_1:
        producer:
          ImageLoader:
            image_path: "D:\\Output\\OpenShot\\Old\\Test_01.mp4"
            loader: Video
            normalized_image_shape: [ 3, 608, 1088 ]

        consumer:
          Track:
            arch: DLA+GhostPAN_mot_8class
            load_model: D:\Project\PythonScripts\Argus\results\train_result\Experiment_02\DLA+GhostPAN_mot_8class\2024-03-08-03-19-33\DLA+GhostPAN_mot_8class.pth
            conf_thres: 0.4
            track_buffer: 30

          PathPredict:
            predictor_name: HermiteSpline

        post:
          IndiResultsVisual:
            output_format: video

        static_shared_value:s
          CamIntrinsicPara:
            data_type: SharedArray_Float
            data_shape: [ 3, 4 ]
            data_value: [ [ 11.11, 0., 128.0, 0 ],
                          [ 0, 11.11, 128.0, 0 ],
                          [ 0, 0, 1.0000, 0 ] ]
      ```
      - First hierarchy is the pipeline name, here is `FuncTest_1`.
      - `producer` contain all producer processes and their own arguments, here we have `ImageLoader`.
      - `consumer` contain all consumer processes and their own arguments, here we have `Track` and `PathPredict`.
      - `post` contain all post processes and their own arguments, here we have `IndiResultsVisual`.
      - All processes are from `multiprocess_pipeline/process`. Check each category `__init__.py` file for process name.
      - `static_shared_value`: for some initial static shared values in this pipeline, here we initialize `CamIntrinsicPara` as camera intrinsic matrix. 
      Check `dict_SharedDataInfoFormat` in `multiprocess_pipeline/shared_structure/__init__.py` for shared value initialize format.

  - **Parser for command-line options**
    
    `opts` handles all the command-line input options or arguments for each experiment. Substructure as following:
    ```
    opts
    ├─── track.py
    ├─── train.py
    └─── _base_opt.py
    ```
    - `_base_opt.py`: master opt class.

    Currently, only 2 optional subclass `opt_track` and `opt_train` are used in this project for multi-object tracking and neural network training respectively.

