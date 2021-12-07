# _Does Knowledge Distillation Really Work?_

This repo contains the code to reproduce experiments performed in [Does Knowledge Distillation Really Work?](https://arxiv.org/abs/2106.05945).

We use [Hydra](https://hydra.cc/) for configuration.
To use Hydra's multirun feature use the `-m` flag (e.g. to run multiple trials use `-m trial_id=0,1,...`).

## Installation

```bash
git clone https://github.com/samuelstanton/gnosis.git
cd gnosis
conda create --name gnosis-env python=3.8
conda activate gnosis-env
python -m pip install -r requirements.txt
python -m pip install -e .
```

#### (Optional) symlink datasets

If you already have the datasets downloaded, just create a symlink.
If you skip this step the datasets will be downloaded automatically.

```bash
mkdir ./data
ln -s <DATASET_PARENT_DIR> ./data/datasets
```


## Train teacher networks

For the sake of efficiency, we recommend you train your teacher and student models separately.

#### Example: train 3 ResNet-20 image classifiers on CIFAR-100

`python scripts/image_classification.py -m teacher.use_ckpts=False classifier.depth=20 
trainer.num_epochs=200 trainer.optimizer.lr=0.1 trainer.lr_scheduler.eta_min=0. 
trainer.distill_teacher=False dataloader.batch_size=256 trial_id=0,1,2`

#### Example: train 3 LSTM text classifiers on IMDB

`python scripts/text_classification.py -m teacher.use_ckpts=False
trainer.distill_teacher=False trial_id=0,1,2`


## (Optional) train a generative model for synthetic data augmentation

To perform synthetic data augmentation you'll first need to train a GAN checkpoint.

`python scripts/image_generation.py`


## Distill student networks

#### Example: distill image classifiers 

`python scripts/image_classification.py -m trial_id=0,1,2 exp_name=student_resnet_baseline_results`

#### Example: distill image classifiers with mixup data augmentation

`python scripts/image_classification.py -m trial_id=0,1,2 
exp_name=student_resnet_mixup_results distill_loader.mixup_alpha=1.`

#### Example: distill image classifiers with synthetic data augmentation (1:4 synthetic to real ratio)

`python scripts/image_classification.py -m trial_id=0,1,2
exp_name=student_resnet_synth-aug_results distill_loader.synth_ratio=0.2`

#### Example: distill text classifiers

`python scripts/text_classification.py -m trial_id=0,1,2 exp_name=student_lstm_baseline_results`


## Logging and Checkpointing

By default, program output and checkpoints are stored locally in automatically generated subdirectories.

To log results to an S3 bucket (must have AWS credentials configured), use
`logger=s3 logger.bucket_name=<BUCKET_NAME>`

To load checkpoints from S3, use
`ckpt_store=s3 s3_bucket=<BUCKET_NAME> teacher.ckpt_path=<TEACHER_REMOTE_PATH> density_model.ckpt_path=<DM_REMOTE_PATH>`


## Additional functionality

Users are encouraged to consult the configuration files in the `config` directory. 
Almost every aspect of the program is configurable from the command line.


## Citation

This project is made freely available under an MIT license. 
If you make use of any part of the code, please cite

```
@article{stanton2021does,
  title={Does Knowledge Distillation Really Work?},
  author={Stanton, Samuel and Izmailov, Pavel and Kirichenko, Polina and Alemi, Alexander A and Wilson, Andrew Gordon},
  journal={arXiv preprint arXiv:2106.05945},
  year={2021}
}
```

The SN-GAN implementation and evaluation is copied from 
[here](https://github.com/mfinzi/olive-oil-ml/blob/master/oil/architectures/img_gen/resnetgan.py).

The CKA implementation is copied from [here](https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment).