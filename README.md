# GGCM
Repository for the paper : 
GGCM: Gradient-Guided for Cross-Domain Few-Shot Learning

If you have any questions/advices/potential ideas, welcome to contact me by huisiqi@stu.xjtu.edu.cn.


# 1 Dependencies
A anaconda envs is recommended:
```
conda create --name py36 python=3.6
conda activate py36
conda install pytorch torchvision -c pytorch
pip3 install scipy>=1.3.2
pip3 install tensorboardX>=1.4
pip3 install h5py>=2.9.0
```

# 2 datasets
We evaluate our methods on five datasets: mini-Imagenet works as source dataset, cub, cars, places, and plantae serve as the target datasets, respectively. 
1. The datasets can be conviently downloaded and processed as in [FWT](https://github.com/hytseng0509/CrossDomainFewShot).
2. Remember to modify your own dataset dir in the 'options.py'.
3. We follow the same the same auxiliary target images as in previous work [meta-FDMixup](https://github.com/lovelyqian/Meta-FDMixup), and the used jsons have been provided in the output dir of this repo.

# 3 pretraining
As in most of the previous CD-FSL methods, a pretrained feature extractor `baseline`   is used.
- you can directly download it from [this link](https://drive.google.com/file/d/1iYu3lvYDixVNPYjmyi0MON8-X3aRN4n2/view), rename it as 399.tar, and put it to the `./output/checkpoints/baseline` 

# 4 Usages
Our method is target set specific, and we take the cub target set under the 5-way 1-shot setting as an example.

1. training for Baseline
```
python train_baseline.py --method Baseline --stop_epoch 400 --modelType Student --target_set cub --name Baseline/CUB/1shot --train_aug --warmup baseline --n_shot 1
```
- DATASET: cub/cars/places/plantae  

2. testing for Baseline
```
python test.py --method Baseline --name  Baselien/CUB/1shot --dataset cub --save_epoch 399 --n_shot 1
```
- DATASET: cub/cars/places/plantae
  
3. training for GGCM
```
python train.py  --method GGCM --aug_rate 0.5 --meta_layers 3 4 --lconsist 5.0  --stop_epoch 400 --modelType Student --target_set cub --name GGCM/CUB/1shot \
--train_aug --warmup baseline --n_shot $shot
```
- DATASET: cub/cars/places/plantae  

4. testing for GGCM
```
python test.py --method GGCM --name  GGCM/CUB/1shot --dataset cub --save_epoch 399 --n_shot 1
```
- DATASET: cub/cars/places/plantae  


# 5 pretrained models
We also provide our pretrained models as follows: (coming soon)

- just take them in the right dir. Take GGCM for the 1-shot as an example, rename it as 399.tar, and move it to the `ouput/checkpoints/GGCM-target-set-cub-1shot/`
