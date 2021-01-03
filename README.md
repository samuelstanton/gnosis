# _Gnosis_
Bayesian Ensemble Distillation

## Quickstart
To use Hydra's multirun feature use the `-m` flag (e.g. to run multiple trials use `-m trial_id=0,1,...`).

### Image Classification
`python scripts/image_classification.py`
#### Important Options
- classifer.depth
- teacher.num_components
- trainer.synth_aug.ratio
- trainer.synth_aug.enabled
- loss
- dataset

### Image Generation
`python scripts/image_generation.py`
#### Important Options
- dataset

## Logging and Checkpointing
To log results to an S3 bucket (must have AWS credentials configured), use
`logger=s3 logger.bucket_name=<BUCKET_NAME>`

To load checkpoints from S3, use
`ckpt_store=s3 s3_bucket=<BUCKET_NAME> teacher.ckpt_path=<TEACHER_REMOTE_PATH> density_model.weight_dir<DM_REMOTE_PATH>`
