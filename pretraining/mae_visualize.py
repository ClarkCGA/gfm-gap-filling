import argparse
import functools
import time
from collections import deque
import pathlib

import torch
import torch.distributed as dist
import torch.optim as optim
import torchvision
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanSquaredError
from torchmetrics.regression import MeanAbsoluteError
import tqdm
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import ConcatDataset

import mae.models_mae
from mae.models_mae import MaskedAutoencoderViT
# from preprocessing.dataset import HLS2Dataset as HLSDataset

import pandas as pd
import os
import json
import random
import numpy as np
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset

from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import Dataset

class CombinedDataset(Dataset):
    def __init__(self, data_path, split="train", num_frames=3, img_size=224, bands=["CDL"], num_hls_bands = 6, cloud_range = [0.01,1.0],
                 normalize=True, training_length=8000,
                 # small trick: as all original values are integers, we make mean values as non-integer
                 # so normalized values have no 0, so that we can mark nodata as 0 because it is a good practice
                 # to fill nodata as mean/median.
                 mean=[495.7316,  814.1386,  924.5740, 2962.5623, 2640.8833, 1740.3031], std=[286.9569, 359.3304, 576.3471, 892.2656, 945.9432, 916.1625], indices=None):
        self.root_dir = pathlib.Path(data_path)
        self.image_dir = self.root_dir.joinpath("chips_filtered/")
        self.cloud_dir = self.root_dir.joinpath("cloud_mask/")
        self.split = split
        self.num_frames = num_frames
        self.mask_position = [2]
        self.img_size = img_size
        self.bands = bands
        self.num_hls_bands = num_hls_bands
        self.training_length = training_length
        if self.split == "train":
            self.cloud_range = cloud_range
        if self.split == "validate":
            self.cloud_range = [0.01,1.0]
        self.normalize = normalize

        self.tif_paths, self.tif_catalog = self._get_tif_paths()
        if self.split == "train":
            self.tif_paths = random.sample(self.tif_paths, training_length)
        self.cloud_paths, self.cloud_catalog = self._get_cloud_paths()

        self.n_cloudpaths = len(self.cloud_paths)
        self.mean = np.array(mean * 3)[:, np.newaxis, np.newaxis]  # corresponding mean per band for normalization purpose
        self.std = np.array(std * 3)[:, np.newaxis, np.newaxis]  # corresponding std per band for normalization purpose

    # Create list of all merged image chips
    def _get_tif_paths(self):
        csv = pd.read_csv(self.root_dir.joinpath("final_chip_tracker.csv"))
        catalog = csv.loc[(csv["usage"] == self.split) & (csv["bad_pct_max"] < 5) & (csv["na_count"] == 0)]
        itemlist = sorted(catalog["chip_id"].tolist())
        pathlist = [self.image_dir.joinpath(f"{item}_merged.tif") for item in itemlist]
        chipslist = list(self.image_dir.glob("*.tif"))
        truelist = sorted(list(set(pathlist) & set(chipslist)))
        sorted_catalog = catalog.sort_values(by="chip_id", ascending=True)
        return truelist, sorted_catalog

    # Create list of all paths to clouds
    def _get_cloud_paths(self):
        csv = pd.read_csv(self.root_dir.joinpath("fmask_tracker_balanced.csv"))
        catalog = csv.loc[(csv["usage"] == self.split) & (csv["cloud_pct"] <= self.cloud_range[1]) & (csv["cloud_pct"] >= self.cloud_range[0])]
        itemlist = sorted(catalog["fmask_name"].tolist())
        chipslist = list(self.cloud_dir.glob("*.tif"))
        pathlist = [self.cloud_dir.joinpath(f"{item}") for item in itemlist]
        truelist = sorted(list(set(pathlist) & set(chipslist)))
        return truelist, catalog
    
    def __len__(self):
        return len(self.tif_paths)

    def __getitem__(self, index):
        def read_tif_as_np_array(path):
            with rasterio.open(path) as src:
                    return src.read()

        # Read in merged tif as ground truth
        groundtruth = read_tif_as_np_array(self.tif_paths[index]) # Need to normalize here

        if self.normalize:
            groundtruth = np.where(groundtruth == -9999, 0.0001,
                                    (groundtruth - self.mean) / self.std)  # don't normalize on nodata
        else:
            groundtruth = groundtruth * 0.0001  # if not normalize, just rescale
        
        # Transpose to bands, time steps, height, width
        groundtruth = groundtruth.reshape(self.num_frames, self.bands, self.img_size, self.img_size)

        # Initialize empty cloud mask with same dimensions as ground truth
        cloudbrick = np.zeros_like(groundtruth)

        # For every specified mask position, read in a random cloud scene and add to the block of cloud masks
        if self.split == "train":
            for p in self.mask_position:
                cloudscene = read_tif_as_np_array(self.cloud_paths[np.random.randint(0,self.n_cloudpaths-1)])
                cloudscene = np.expand_dims(cloudscene, 0)
                cloudbrick[p-1,:,:,:] = cloudscene # Check if this works, the code should assign cloud scene to ALL these values in the 4 channels indexed.
                del cloudscene

        if self.split == "validate":
            for p in self.mask_position:
                cloudscene = read_tif_as_np_array(self.cloud_paths[index % self.n_cloudpaths])
                cloudscene = np.expand_dims(cloudscene, 0)
                cloudbrick[p-1,:,:,:] = cloudscene # Check if this works, the code should assign cloud scene to ALL these values in the 4 channels indexed.
                del cloudscene

        cloudbrick = np.expand_dims(cloudbrick, 0) # Adds a dimension at index 0
        groundtruth = np.expand_dims(groundtruth, 0) 
        # Concatenate the tensors along the new dimension
        combined_data = np.concatenate((groundtruth, cloudbrick), axis=0).astype(np.float32).transpose(0,2,1,3,4)

        # A tensor with dimensions (mask-or-image, bands, time steps, height, width)
        return combined_data

def visualize_tcc(vis_path, idx, input, input_mask, processed_mask, predicted):

    n_timesteps = input.size()[2]
    input_list = []
    reconstruction_list = []
    groundtruth_list = []
    "torch.Size([1, 6, 3, 224, 224])"
    for t in range(1, n_timesteps+1):
        input_img = input[0,0:3,t-1,:,:].clone().flip(0) * 3
        input_mask_img = input_mask[0,0:3,t-1,:,:].clone()
        input_masked = torch.where(input_mask_img == 1, input_mask_img, input_img)
        input_masked = torch.nn.functional.pad(input_masked, (2,2,2,2), value=0)
        input_list.append(input_masked)
    for t in range(1, n_timesteps+1):
        processed_mask_img = input_mask[0,0:3,t-1,:,:].clone()
        predicted_img = predicted[0,0:3,t-1,:,:].clone().flip(0) * 3
        input_img = input[0,0:3,t-1,:,:].clone().flip(0) * 3
        predicted_unmasked = predicted_img * processed_mask_img
        input_masked = input_img * (1-processed_mask_img)
        reconstructed_img = predicted_unmasked + input_masked
        reconstructed_img = torch.nn.functional.pad(reconstructed_img, (2,2,2,2), value=0)
        reconstruction_list.append(reconstructed_img)
    for t in range(1, n_timesteps+1):
        input_img = input[0,0:3,t-1,:,:].clone().flip(0) * 3
        input_img = torch.nn.functional.pad(input_img, (2,2,2,2), value=0)
        groundtruth_list.append(input_img)
    input_list = torch.cat(input_list, dim=1)
    reconstruction_list = torch.cat(reconstruction_list, dim=1)
    groundtruth_list = torch.cat(groundtruth_list, dim=1)
    full_vis = torch.cat([input_list]+[reconstruction_list]+[groundtruth_list], dim=2)
    torchvision.utils.save_image(
        full_vis, os.path.join(vis_path, "idx{:04}_gen.jpg".format(idx)),
    )

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    # data loader related
    parser.add_argument('--train_dir', default='/dev/shm/train', type=str,
                        help='Path to the train data directory.')
    parser.add_argument('--val_dir', default='/dev/shm/val', type=str,
                        help='Path to the validation data directory.')
    parser.add_argument('--mask_dir', default='/dev/shm/train', type=str,
                        help='Path to the mask data directory.')
    parser.add_argument('--num_frames', default=3, type=int,
                        help='Number of frames in a sample.')
    parser.add_argument('--img_size', default=224, type=int,
                        help='Input image size.')
    parser.add_argument('--bands', default=["B02", "B03", "B04", "B05"], type=str, nargs='+',
                        help='Spectral bands to use.',
                        choices=["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11"])
    parser.add_argument('--random_cropping', action='store_true',
                        help='Use random cropping for input data. Default = True')
    parser.add_argument('--no_random_cropping', action='store_false', dest='random_cropping')
    parser.set_defaults(random_cropping=True)
    parser.add_argument('--data_loader_num_workers', default=2, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--training_length', default=8000, type=int,
                        help='Number of training chips.')

    parser.add_argument(
        "--cloud_range",
        type=float,
        default=[0,1],
        nargs="+",
        help="Lower and upper boundaries for cloud ratios",
    )
    # model related
    parser.add_argument('--num_layers', default=12, type=int,
                        help='Number of layers in the model.')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='Input patch size.')
    parser.add_argument('--embed_dim', default=768, type=int,
                        help='Number of embeddings dimensions.')
    parser.add_argument('--num_heads', default=8, type=int,
                        help='Number of heads in the model.')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--tubelet_size', default=1, type=int,
                        help='Temporal patch size.')
    parser.add_argument('--checkpoint', default='/workspace/gfm-gap-filling/pretraining/epoch-832-loss-0.0473.pt', type=str,
                        help='Path to a checkpoint file to load from.')
    parser.add_argument('--job_id', default='default', type=str,
                        help='Path to a job id to load from.')

    # training related
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate.')
    # LR decay not being used
    # parser.add_argument('--lr_decay', type=float, default=0.85,
    #                     help='Learning rate decay')
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--local_rank', default=0, type=int)

    # logging related
    parser.add_argument('--base_log_dir', default='/workspace/data/lchu/hls/logs',
                        help='Path to the root directory where to save log files.')
    parser.add_argument('--base_checkpoint_dir', default='/workspace/data/lchu/hls/checkpoints',
                        help='Path to root directory where to save checkpoints.')
    parser.add_argument('--base_visualization_dir', default='/workspace/data/lchu/hls/vis',
                        help='Path to the root directory where to save visualizations.')
    parser.add_argument('--base_job_info_dir', default='/workspace/data/lchu/hls/jobs',
                        help='Path to the root directory where to save job info file.')

    return parser

def validation(model, mask_ratio, local_rank, rank, test_loader, n_epoch, vis_path):
    model.eval()
    ddp_loss = torch.zeros(2).to(local_rank)
    mask_ratio = torch.zeros(2).to(local_rank)
    ssim = torch.zeros(2).to('cpu')
    mse = torch.zeros(2).to('cpu')
    mae = torch.zeros(2).to('cpu')
    mean = torch.tensor([495.7316,  814.1386,  924.5740, 2962.5623, 2640.8833, 1740.3031])[None,:,None,None,None].to('cpu')
    std = torch.tensor([286.9569, 359.3304, 576.3471, 892.2656, 945.9432, 916.1625])[None,:,None,None,None].to('cpu')
    inner_pbar = tqdm.tqdm(
        range(len(test_loader)), colour="green", desc="Validation Epoch", leave=True
    )
    
    StructuralSimilarity = StructuralSimilarityIndexMeasure(data_range=1.0)
    mean_squared_error = MeanSquaredError()
    mean_abs_error = MeanAbsoluteError()

    data_list = [] 

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            label_mask_batch = batch[:,1,:,:,:,:]

            batch = batch[:,0,:,:,:,:]
          #  label_mask_batch = mask_val_loader_list[i]
            loss, pred, mask = model(batch.to(local_rank), label_mask_batch.to(local_rank), mask_ratio)

            mask_ratio[0] += torch.mean(mask) * 3 # Adjust to only one mask position
            mask_ratio[1] += 1
            
            ddp_loss[0] += loss.item()  # sum up batch loss
            ddp_loss[1] += 1

            inner_pbar.update(1)
            
            input = (batch.detach().cpu() * std + mean) * 0.0001
            input_mask = label_mask_batch.detach().cpu()
            processed_mask = model.unpatchify(mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1])).detach().cpu()
            predicted = (model.unpatchify(pred).detach().cpu() * std + mean) * 0.0001
            input_masked = input * input_mask
            predicted_masked = predicted * input_mask

            if i % 80 == 0:
                visualize_tcc(vis_path, i, input, input_mask, processed_mask, predicted)
            
            ssim_score = StructuralSimilarity(predicted_masked[:,:,1,:,:], input_masked[:,:,1,:,:])

            ssim[0] += ssim_score
            ssim[1] += 1
  
            mse_score = mean_squared_error(predicted_masked, input_masked)
            mse_score /= (torch.mean(mask).detach().cpu())
            
            mse[0] += mse_score
            mse[1] += 1

            mae_score = mean_abs_error(predicted_masked[:,:,1,:,:], input_masked[:,:,1,:,:])
            mae_score /= (torch.mean(processed_mask[:,:,1,:,:]))
        
            mae[0] += mae_score
            mae[1] += 1

            per_band_mse_list = []
            per_band_mae_list = []
            per_band_ssim_list = []

            for n in range(6): # For each band of 6, do the following:
                # Get the MSE for only that band, selected with n, from the predicted and input, masked with the cloud mask.
                per_band_mse = mean_squared_error(predicted_masked[:,n:n+1,:,:,:], input_masked[:,n:n+1,:,:,:])
                # Adjust the mse by the proportion of masked pixels.
                per_band_mse /= (torch.mean(mask).detach().cpu())
                # Append to the list of per band mse for this batch
                per_band_mse_list.append(per_band_mse.item())
                # Get the MAE for only that band, selected with n, from the predicted and input, masked with the cloud mask.
                per_band_mae = mean_abs_error(predicted_masked[:,n:n+1,:,:,:], input_masked[:,n:n+1,:,:,:])
                # Adjust the mae by the proportion of masked pixels.
                per_band_mae /= (torch.mean(mask).detach().cpu())
                # Append to the list of per band mae for this batch
                per_band_mae_list.append(per_band_mae.item())
                # Get the SSIM for only that band at the middle time step.
                per_band_ssim_score = StructuralSimilarity(predicted_masked[:,n:n+1,1,:,:], input_masked[:,n:n+1,1,:,:])
                # Append to the list of per band SSIM for this batch
                per_band_ssim_list.append(per_band_ssim_score.item())

            # Append a dictionary representing this batch's stats, to be compiled into a dataframe
            data_list.append({'Overall SSIM':ssim_score.item(), 
                              'Overall MSE':mse_score.item(),
                              'Overall MAE':mae_score.item(),
                              'Mask Ratio':torch.mean(mask).item() * 3,
                              'B02 MSE': per_band_mse_list[0],
                              'B03 MSE': per_band_mse_list[1],
                              'B04 MSE': per_band_mse_list[2],
                              'B05 MSE': per_band_mse_list[3],
                              'B07 MSE': per_band_mse_list[4],
                              'B08 MSE': per_band_mse_list[5],
                              'B02 MAE': per_band_mae_list[0],
                              'B03 MAE': per_band_mae_list[1],
                              'B04 MAE': per_band_mae_list[2],
                              'B05 MAE': per_band_mae_list[3],
                              'B07 MAE': per_band_mae_list[4],
                              'B08 MAE': per_band_mae_list[5],
                              'B02 SSIM': per_band_ssim_list[0],
                              'B03 SSIM': per_band_ssim_list[1],
                              'B04 SSIM': per_band_ssim_list[2],
                              'B05 SSIM': per_band_ssim_list[3],
                              'B07 SSIM': per_band_ssim_list[4],
                              'B08 SSIM': per_band_ssim_list[5],
                             })
            
                              
            

        # dist.barrier()

    # dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    val_loss = ddp_loss[0] / ddp_loss[1]
    epoch_mask_ratio = mask_ratio[0] / mask_ratio[1]
    epoch_ssim = ssim[0] / ssim[1]
    epoch_mse = mse[0] / mse[1]

    inner_pbar.close()
    print(f"Validation Loss: {val_loss:.4f}, Mask Ratio: \t{epoch_mask_ratio:.4f}, SSIM: {epoch_ssim:.4f}, MSE: {epoch_mse:.4f}")
    return val_loss, epoch_mask_ratio, epoch_ssim, epoch_mse, data_list


def fsdp_main(args):
    """main process, run within each individual GPU process"""
    # TODO: can we get the time the job was submitted?
    start_time = f"{time.strftime('%Y-%m-%d %H:%M:%S')}"

    # debug nan gradient
    torch.autograd.set_detect_anomaly(True)

    # torchrun specific
    local_rank = args.local_rank
    rank = 0
    world_size = 1

    ### Configs
    # data loader related
    train_dir = args.train_dir
    val_dir = args.val_dir
    mask_dir = args.mask_dir
    num_frames = args.num_frames
    img_size = args.img_size
    bands = args.bands
    num_hls_bands = len(bands)
    cloud_range = args.cloud_range
    training_length = args.training_length
   # random_cropping = args.random_cropping
    num_workers = args.data_loader_num_workers

    # model related
    num_layers = args.num_layers
    patch_size = args.patch_size
    embed_dim = args.embed_dim
    num_heads = args.num_heads
    mask_ratio = args.mask_ratio
    tubelet_size = args.tubelet_size
    checkpoint = args.checkpoint
    job_id = args.job_id

    # training related
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.num_epochs

    ckpt_dir = os.path.join(args.base_checkpoint_dir, job_id)
    vis_dir = os.path.join(args.base_visualization_dir, job_id)
    job_info_dir = os.path.join(args.base_job_info_dir)
    tensorboard_log_dir = os.path.join(args.base_log_dir, "tensorboard", job_id)
    csv_log_dir = os.path.join(args.base_log_dir, "csv", job_id)

    # save job info in a yaml file
    params_dict = dict(vars(args))

        # Add more info
    params_dict['job_id'] = job_id
    params_dict['checkpoint_dir'] = ckpt_dir
    params_dict['visualization_dir'] = vis_dir
    params_dict['tensorboard_dir'] = tensorboard_log_dir
    params_dict['csv_dir'] = csv_log_dir
    params_dict['world_size'] = world_size
    params_dict['job_start_time'] = start_time
    params_dict['job_finish_time'] = 'NA'

    os.makedirs(job_info_dir, exist_ok=True)
    os.makedirs(csv_log_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    with open(os.path.join(job_info_dir, f'{job_id}_visualization.yaml'), 'w') as f:
        yaml.safe_dump(params_dict, f, default_flow_style=None, sort_keys=False)

    def prepare_model(checkpoint, arch='mae_vit_base_patch16'):
        # build model
        model = getattr(mae.models_mae, arch)()
        # load model
        checkpoint_file = torch.load(checkpoint, map_location=f'cuda:{local_rank}')
        msg = model.load_state_dict(checkpoint_file, strict=False)
        print(msg)
        return model

    model = prepare_model(checkpoint, 'mae_vit_base_patch16')
    print('Model loaded.')

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> model has {total_params / 1e6} Million params.\n")

    val_dataset = CombinedDataset(train_dir, split="validate", num_frames=3, img_size=224, bands=6, cloud_range=cloud_range,
                              # random_cropping=random_cropping, remove_cloud=True, 
                               normalize=True)
    if rank == 0:
        print(f"--> Validation set len = {len(val_dataset)}")
    if rank == 0:
        print(f"--> Validation set masks = {val_dataset.n_cloudpaths}")

    val_chip_dataframe = pd.DataFrame(val_dataset.tif_catalog)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    test_kwargs = {"batch_size": batch_size, "sampler": val_sampler}
    common_kwargs = {
        #"num_workers": num_workers,
        "pin_memory": False,
        "drop_last": True
    }

    test_kwargs.update(common_kwargs)

    test_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)

    torch.cuda.set_device(local_rank)

    torch.cuda.empty_cache()

    model = model.to(torch.cuda.current_device())

    # -- Start Training -----
    for epoch in range(1, epochs + 1):

        curr_val_loss, val_mask_ratio, val_ssim, val_mse, data_list = validation(model, mask_ratio, local_rank, rank, test_loader, epoch, vis_path=vis_dir)
        stats_df = pd.DataFrame(data_list)
        chip_stats_df = pd.concat([val_chip_dataframe.reset_index(drop=True), stats_df.reset_index(drop=True)], axis=1)
        chip_stats_df.to_csv(os.path.join(csv_log_dir, "chip_stats.csv"), index=False)


    # all done, set barrier to ensure all GPU's complete, and then cleanup
    # dist.barrier()
    # dist.destroy_process_group()


# ------------------ Main functions above ------------


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    fsdp_main(args)

