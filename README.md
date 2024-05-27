# gfm-gap-filling
Fine-tuning the GFM model for gap filling task. 

Original pretraining script https://github.com/ClarkCGA/hls-foundation/blob/main/hls/pretraining/mae_training.py

MAE creation script https://github.com/facebookresearch/mae/blob/main/models_mae.py#

MAE overview https://huggingface.co/docs/transformers/model_doc/vit_mae  

## Directory Structure

Directories should be structured as follows:

```
gfm/
├── gfm-gap-filling/
│ ├── pretraining/
│ │ ├── training_data/
│ ├── Dockerfile
│ ├── .git
│ ├── README.md
├── data/
  ├── training_data/
```

## Running the Docker Image

Navigate to directory `gfm` and run 
```
docker run --gpus all -it -v $PWD:/workspace/ -p 8888:8888 gapfill
```
This will start a Jupyer Lab session which can be accessed via a web browser.

## Running the Fine-Tuning Script

In order to run the fine-tuning script, run mae_training.py on the command line as a python module. For example:


```
python -m mae_training --train_dir "/workspace/data/training_data" --batch_size 16 --num_epochs 200 --embed_dim 768 --cloud_range 0.01 1.0 --mask_position 2 --training_len 400 --local_rank 0 
```
**--train_dir** should point to the data directory containing .csv chip trackers and subfolders for hls data and cloud masks.

**--num_epochs** is the number of epochs the model will train for.

**--batch_size** can be modified according to memory constraints.

**--embed_dim** must be 768 to use the pretrained weights, but can be modified if training from scratch.

**--cloud_range** is the lower and upper limits of the ratio of clouds for masks that will be input randomly during training. During validation, the same set of cloud masks are used regardless of inputs for testing consistency across experiments.

**--local_rank** determines which GPU the module will run on. This allows for parallel experiments on machines with multiple GPUs available.

**--mask_position** defines which combinations of time steps will be masked. For example, an input of --mask_position 12 23 123 would cause the training to rotate between masking time step 1 and 2, 2 and 3, and 1, 2, and 3.

**--training_len** defines the number of time series image chips the model will train on. These will be randomly subsampled from the training set.

## Generating Graphs of Training Performance

Use create_graphs.ipynb to create graphs of model performance during fine-tuning. Replace the variable `job_id` with the experiment whose performance you want to visualize, e.g. `6231-fair-bs16-2023-08-21_16-49-34`

## Generating Example Images and Per-Image Statistics

This can be run for any weights checkpoint, including the pretrained weights. The script will access a checkpoint and save images to the visualization directory assocciated with the job id that is passed in the command line, creating it if it does not exist. For zero-shot, run as follows:

```
python -m mae_visualize --train_dir "/workspace/gfm-gap-filling/pretraining/training_data" --batch_size 1 --num_epochs 1 --embed_dim 768 --cloud_range 0.01 1.0 --local_rank 0 --checkpoint /workspace/gfm-gap-filling/pretraining/epoch-832-loss-0.0473.pt --job_id  zero_shot
```

For visualizing the results of fine tuning, run as follows:

```
python -m mae_visualize --train_dir "/workspace/gfm-gap-filling/pretraining/training_data" --batch_size 1 --num_epochs 1 --embed_dim 768 --cloud_range 0.01 1.0 --local_rank 0 --checkpoint /workspace/data/lchu/hls/checkpoints/6231-fair-bs16-2023-08-21_16-49-34/model_best.pt --job_id  6231-fair-bs16-2023-08-21_16-49-34
```

**--batch_size** must be 1 in order to extract statistics from ALL testing images

## Generating Visualizations of Per-Image Statistics

Use per_image_graphs.ipynb to create visualizations of the distributions and correlations of per-image performance metrics. Replace the variable `job_id` with the experiment whose performance you want to visualize, e.g. `6231-fair-bs16-2023-08-21_16-49-34`

## Generating Visualizations of Band Correlations

Use band_correlations.ipynb to create visualizations of band correlations for the low-coverage example image and the first 200 images of testing. Replace the variable `job_id` with the experiment whose performance you want to visualize, e.g. `6231-fair-bs16-2023-08-21_16-49-34`
