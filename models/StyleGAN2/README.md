
# StyleGAN2-Jittor
A Jittor Implementation of StyleGAN2.

## Results
Below are samples of human faces generated with intermediate trained model of StyleGAN2-Jittor, trained on the FFHQ dataset with 128 resolution.
![sample_image](https://github.com/zhouwy19/XNN-Project/blob/main/StyleGAN2-Jittor/samples/sample.png)

## Set up Environment
Please install the required packages by running:
```
pip install -r requirements.txt
```

## Prepare Datasets
Preprocessing of datasets is also adapted (and simplified) from NVIDIA's official implementation. 
Please run the following to generate a dataset json file for training.
```
python training/dataset.py --source "DATA_PATH" --save-path "SAVE_PATH" 
```
For example, for processing the FFHQ dataset, you may want to run:
```
python training/dataset.py --source "./FFHQ" \ # please replace with path to local FFHQ folder
                        --save-path "./dataset" 
```
After preprocessing is complete, the save-path will contain a dataset.json file containing critical information. If preprocessing is somehow interrupted, note that dataset.py requires the target folder to be completely empty to function properly.  

## Training
To train a model, run 
```
python train.py --data_path DATAPATH --ckpt_path PATH_TO_SAVE_TRAINED_MODEL
```

## Evaluation
To evaluate the trained model with FID score, tun
```
python train.py --data_path DATAPATH --eval_path EVAL_IMAGE_PATH --ckpt_path PATH_TO_SAVE_TRAINED_MODEL
```

## Acknowledgements
This repository is based on and modified from [Nvidia PyTorch implementation of StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch.git), [StyleGAN 2 in PyTorch](https://github.com/rosinality/stylegan2-pytorch.git), [Jittor Version of StyleGAN3 (Alias-Free Generative Adversarial Networks)](https://github.com/ty625911724/Jittor_StyleGAN3.git), and [GAN_Sketch](https://github.com/KIMI-debug-maker/GAN_Sketch.git).
