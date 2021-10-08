# Training and evaluation 


## Table of contents
* [3D hand experiment](#3D-hand-reconstruction-from-a-single-image)
    * [Training](#Training)
    * [Testing](#Testing)
* [3D human body experiment](#Human-mesh-reconstruction-from-a-single-image)
    * [Training with mixed 2D+3D datasets](#Training-with-mixed-datasets)
    * [Evaluation on Human3.6M](#Evaluation-on-Human3.6M)
    * [Training with 3DPW](#Training-with-3DPW-dataset)
    * [Evaluation on 3DPW](#Evaluation-on-3DPW)


## 3D hand reconstruction from a single image

### Training

We use the following script to train on FreiHAND dataset. 

```bash
python -m torch.distributed.launch --nproc_per_node=8 \
       src/tools/run_gphmer_handmesh.py \
       --train_yaml freihand/train.yaml \
       --val_yaml freihand/test.yaml \
       --arch hrnet-w64 \
       --num_workers 4 \
       --per_gpu_train_batch_size 32 \
       --per_gpu_eval_batch_size 32 \
       --num_hidden_layers 4 \
       --num_attention_heads 4 \
       --lr 1e-4 \
       --num_train_epochs 200 \
       --input_feat_dim 2051,512,128 \
       --hidden_feat_dim 1024,256,64 
```


Example training log can be found here [2021-03-06-graphormer_freihand_log.txt](https://datarelease.blob.core.windows.net/metro/models/2021-03-06-graphormer_freihand_log.txt)

### Testing

After training, we use the final checkpoint (trained at 200 epoch) for testing.

We use the following script to generate predictions. It will generate a prediction file called `ckpt200-sc10_rot0-pred.zip`. Afte that, please submit the prediction file to [FreiHAND Leaderboard](https://competitions.codalab.org/competitions/21238) to obtain the evlauation scores.


In the following script, we perform prediction with test-time augmentation on FreiHAND experiments. We will generate a prediction file `ckpt200-multisc-pred.zip`. 

```bash
python src/tools/run_hand_multiscale.py \
       --multiscale_inference \
       --model_path models/graphormer_release/graphormer_hand_state_dict.bin \
```

To reproduce our results, we have released our prediction file `ckpt200-multisc-pred.zip` (see `docs/DOWNLOAD.md`). You may want to submit it to the Leaderboard, and it should produce the following results. 

```bash
Evaluation 3D KP results:
auc=0.000, mean_kp3d_avg=71.48 cm
Evaluation 3D KP ALIGNED results:
auc=0.883, mean_kp3d_avg=0.59 cm

Evaluation 3D MESH results:
auc=0.000, mean_kp3d_avg=71.47 cm
Evaluation 3D MESH ALIGNED results:
auc=0.880, mean_kp3d_avg=0.60 cm

F-scores
F@5.0mm = 0.000 	F_aligned@5.0mm = 0.764
F@15.0mm = 0.000 	F_aligned@15.0mm = 0.986
```

Note that our method predicts relative coordinates (there is no global alignment). Therefore, only the aligned scores are meaningful in our case.


## Human mesh reconstruction from a single image


### Training with mixed datasets

We conduct large-scale training on multiple 2D and 3D datasets, including Human3.6M, COCO, MUCO, UP3D, MPII. During training, it will evaluate the performance per epoch, and save the best checkpoints.

```bash
python -m torch.distributed.launch --nproc_per_node=8 \
       src/tools/run_gphmer_bodymesh.py \
       --train_yaml Tax-H36m-coco40k-Muco-UP-Mpii/train.yaml \
       --val_yaml human3.6m/valid.protocol2.yaml \
       --arch hrnet-w64 \
       --num_workers 4 \
       --per_gpu_train_batch_size 25 \
       --per_gpu_eval_batch_size 25 \
       --num_hidden_layers 4 \
       --num_attention_heads 4 \
       --lr 1e-4 \
       --num_train_epochs 200 \
       --input_feat_dim 2051,512,128 \
       --hidden_feat_dim 1024,256,64
```

Example training log can be found here [2021-02-25-graphormer_h36m_log](https://datarelease.blob.core.windows.net/metro/models/2021-02-25-graphormer_h36m_log.txt)

### Evaluation on Human3.6M

In the following script, we evaluate our model `graphormer_h36m_state_dict.bin` on Human3.6M validation set. Check `docs/DOWNLOAD.md` for more details about downloading the model file.

```bash
python -m torch.distributed.launch --nproc_per_node=8 \
          src/tools/run_gphmer_bodymesh.py \
          --val_yaml human3.6m/valid.protocol2.yaml \
          --arch hrnet-w64 \
          --num_workers 4 \
          --per_gpu_eval_batch_size 25 \
          --num_hidden_layers 4 \
          --num_attention_heads 4 \
          --input_feat_dim 2051,512,128 \
          --hidden_feat_dim 1024,256,64 \
          --run_eval_only \
          --resume_checkpoint ./models/graphormer_release/graphormer_h36m_state_dict.bin 
```

We show the example outputs of this script as below. 
```bash
2021-09-19 13:18:14,416 Graphormer INFO: Using 8 GPUs
2021-09-19 13:18:18,712 Graphormer INFO: Update config parameter num_hidden_layers: 12 -> 4
2021-09-19 13:18:18,718 Graphormer INFO: Update config parameter hidden_size: 768 -> 1024
2021-09-19 13:18:18,725 Graphormer INFO: Update config parameter num_attention_heads: 12 -> 4
2021-09-19 13:18:18,731 Graphormer INFO: Update config parameter intermediate_size: 3072 -> 2048
2021-09-19 13:18:19,983 Graphormer INFO: Init model from scratch.
2021-09-19 13:18:19,990 Graphormer INFO: Update config parameter num_hidden_layers: 12 -> 4
2021-09-19 13:18:19,995 Graphormer INFO: Update config parameter hidden_size: 768 -> 256
2021-09-19 13:18:20,001 Graphormer INFO: Update config parameter num_attention_heads: 12 -> 4
2021-09-19 13:18:20,006 Graphormer INFO: Update config parameter intermediate_size: 3072 -> 512
2021-09-19 13:18:20,210 Graphormer INFO: Init model from scratch.
2021-09-19 13:18:20,217 Graphormer INFO: Add Graph Conv
2021-09-19 13:18:20,223 Graphormer INFO: Update config parameter num_hidden_layers: 12 -> 4
2021-09-19 13:18:20,228 Graphormer INFO: Update config parameter hidden_size: 768 -> 64
2021-09-19 13:18:20,233 Graphormer INFO: Update config parameter num_attention_heads: 12 -> 4
2021-09-19 13:18:20,239 Graphormer INFO: Update config parameter intermediate_size: 3072 -> 128
2021-09-19 13:18:20,295 Graphormer INFO: Init model from scratch.
2021-09-19 13:18:23,797 Graphormer INFO: => loading hrnet-v2-w64 model
2021-09-19 13:18:23,805 Graphormer INFO: Graphormer encoders total parameters: 83318598
2021-09-19 13:18:23,814 Graphormer INFO: Backbone total parameters: 128059944
2021-09-19 13:18:23,892 Graphormer INFO: Loading state dict from checkpoint _output/graphormer_release/graphormer_h36m_state_dict.bin
2021-09-19 13:19:26,299 Graphormer INFO: Validation epoch: 0  mPVE:   0.00, mPJPE:  51.20, PAmPJPE:  34.55 
```
 


### Training with 3DPW dataset

We follow prior works that also use 3DPW training data. In order to make the training faster, we **fine-tune** our pre-trained model (`graphormer_h36m_state_dict.bin`) on 3DPW training set. 

We use the following script for fine-tuning. During fine-tuning, it will evaluate the performance per epoch, and save the best checkpoints. 

```bash
python -m torch.distributed.launch --nproc_per_node=8 \
       src/tools/run_gphmer_bodymesh.py \
       --train_yaml 3dpw/train.yaml \
       --val_yaml 3dpw/test_has_gender.yaml \
       --arch hrnet-w64 \
       --num_workers 4 \
       --per_gpu_train_batch_size 20 \
       --per_gpu_eval_batch_size 20 \
       --num_hidden_layers 4 \
       --num_attention_heads 4 \
       --lr 1e-4 \
       --num_train_epochs 5 \
       --input_feat_dim 2051,512,128 \
       --hidden_feat_dim 1024,256,64 \
       --resume_checkpoint  {YOUR_PATH/state_dict.bin} \
```


### Evaluation on 3DPW
In the following script, we evaluate our model `graphormer_3dpw_state_dict.bin` on 3DPW test set. Check `docs/DOWNLOAD.md` for more details about downloading the model file.


```bash
python -m torch.distributed.launch --nproc_per_node=8 \
          src/tools/run_gphmer_bodymesh.py \
          --val_yaml 3dpw/test.yaml \
          --arch hrnet-w64 \
          --num_workers 4 \
          --per_gpu_eval_batch_size 25 \
          --num_hidden_layers 4 \
          --num_attention_heads 4 \
          --input_feat_dim 2051,512,128 \
          --hidden_feat_dim 1024,256,64 \
          --run_eval_only \
          --resume_checkpoint ./models/graphormer_release/graphormer_3dpw_state_dict.bin  
```

After evaluation, it should reproduce the results below
```bash
2021-09-20 00:54:46,178 Graphormer INFO: Using 8 GPUs
2021-09-20 00:54:50,339 Graphormer INFO: Update config parameter num_hidden_layers: 12 -> 4
2021-09-20 00:54:50,345 Graphormer INFO: Update config parameter hidden_size: 768 -> 1024
2021-09-20 00:54:50,351 Graphormer INFO: Update config parameter num_attention_heads: 12 -> 4
2021-09-20 00:54:50,357 Graphormer INFO: Update config parameter intermediate_size: 3072 -> 2048
2021-09-20 00:54:51,602 Graphormer INFO: Init model from scratch.
2021-09-20 00:54:51,613 Graphormer INFO: Update config parameter num_hidden_layers: 12 -> 4
2021-09-20 00:54:51,625 Graphormer INFO: Update config parameter hidden_size: 768 -> 256
2021-09-20 00:54:51,646 Graphormer INFO: Update config parameter num_attention_heads: 12 -> 4
2021-09-20 00:54:51,652 Graphormer INFO: Update config parameter intermediate_size: 3072 -> 512
2021-09-20 00:54:51,855 Graphormer INFO: Init model from scratch.
2021-09-20 00:54:51,862 Graphormer INFO: Add Graph Conv
2021-09-20 00:54:51,868 Graphormer INFO: Update config parameter num_hidden_layers: 12 -> 4
2021-09-20 00:54:51,873 Graphormer INFO: Update config parameter hidden_size: 768 -> 64
2021-09-20 00:54:51,880 Graphormer INFO: Update config parameter num_attention_heads: 12 -> 4
2021-09-20 00:54:51,885 Graphormer INFO: Update config parameter intermediate_size: 3072 -> 128
2021-09-20 00:54:51,948 Graphormer INFO: Init model from scratch.
2021-09-20 00:54:55,564 Graphormer INFO: => loading hrnet-v2-w64 model
2021-09-20 00:54:55,572 Graphormer INFO: Graphormer encoders total parameters: 83318598
2021-09-20 00:54:55,580 Graphormer INFO: Backbone total parameters: 128059944
2021-09-20 00:54:55,655 Graphormer INFO: Loading state dict from checkpoint _output/graphormer_release/graphormer_3dpw_state_dict.bin
2021-09-20 00:56:24,334 Graphormer INFO: Validation epoch: 0  mPVE:  87.57, mPJPE:  73.98, PAmPJPE:  45.35 
```

