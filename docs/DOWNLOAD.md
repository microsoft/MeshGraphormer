# Download

### Getting Started

1. Create folders that store pretrained models, datasets, and predictions.
    ```bash
    export REPO_DIR=$PWD
    mkdir -p $REPO_DIR/models  # pre-trained models
    mkdir -p $REPO_DIR/datasets  # datasets
    mkdir -p $REPO_DIR/predictions  # prediction outputs
    ```

2. Download pretrained models.

    Our pre-trained models can be downloaded with the following command.
    ```bash
    cd $REPO_DIR
    bash scripts/download_models.sh
    ```
    The scripts will download three models that are trained for mesh reconstruction on Human3.6M, 3DPW, and FreiHAND, respectively. For your convenience, this script will also download HRNet pre-trained weights, which will be used in training. 

    The resulting data structure should follow the hierarchy as below. 
    ```
    ${REPO_DIR}  
    |-- models  
    |   |-- graphormer_release
    |   |   |-- graphormer_h36m_state_dict.bin
    |   |   |-- graphormer_3dpw_state_dict.bin
    |   |   |-- graphormer_hand_state_dict.bin
    |   |-- hrnet
    |   |   |-- hrnetv2_w40_imagenet_pretrained.pth
    |   |   |-- hrnetv2_w64_imagenet_pretrained.pth
    |   |   |-- cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
    |   |   |-- cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
    |-- src 
    |-- datasets 
    |-- predictions 
    |-- README.md 
    |-- ... 
    |-- ... 
    ```

3. Download SMPL and MANO models from their official websites

    To run our code smoothly, please visit the following websites to download SMPL and MANO models. 

    - Download `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from [SMPLify](http://smplify.is.tue.mpg.de/), and place it at `${REPO_DIR}/src/modeling/data`.
    - Download `MANO_RIGHT.pkl` from [MANO](https://mano.is.tue.mpg.de/), and place it at `${REPO_DIR}/src/modeling/data`.

    Please put the downloaded files under the `${REPO_DIR}/src/modeling/data` directory. The data structure should follow the hierarchy below. 
    ```
    ${REPO_DIR}  
    |-- src  
    |   |-- modeling
    |   |   |-- data
    |   |   |   |-- basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
    |   |   |   |-- MANO_RIGHT.pkl
    |-- models
    |-- datasets
    |-- predictions
    |-- README.md 
    |-- ... 
    |-- ... 
    ```
    Please check [/src/modeling/data/README.md](../src/modeling/data/README.md) for further details.

4. Download prediction files that were evaluated on FreiHAND Leaderboard.

    The prediction files can be downloaded with the following command.
    ```bash
    cd $REPO_DIR
    bash scripts/download_preds.sh
    ```
    You could submit the prediction files to FreiHAND Leaderboard and reproduce our results.

5. Download datasets and pseudo labels for training.

    We use the same data from our previous project [METRO](https://github.com/microsoft/MeshTransformer)

    Please visit our previous project page to download datasets and annotations for experiments. Click [LINK](https://github.com/microsoft/MeshTransformer/blob/main/docs/DOWNLOAD.md) here.
