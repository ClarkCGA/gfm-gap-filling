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
    def __init__(self, data_path, 
                 split="train", 
                 num_frames=3, 
                 img_size=224, 
                 bands=["CDL"], 
                 num_hls_bands = 6, 
                 cloud_range = [0.01,1.0],
                 normalize=True, 
                 training_length=6321,
                 mean=[495.7316,  814.1386,  924.5740, 2962.5623, 2640.8833, 1740.3031], 
                 std=[286.9569, 359.3304, 576.3471, 892.2656, 945.9432, 916.1625]):
        
        # get all directories needed for reading in chips
        self.root_dir = pathlib.Path(data_path)
        self.image_dir = self.root_dir.joinpath("chips_filtered/")
        self.cloud_dir = self.root_dir.joinpath("cloud_mask/")

        # set parameters
        self.split = split
        self.num_frames = num_frames
        self.mask_position = [[1],[2],[3],[1,2],[2,3],[1,3],[1,2,3]]
        self.img_size = img_size
        self.bands = bands
        self.num_hls_bands = num_hls_bands
        self.training_length = training_length
        self.normalize = normalize

        # ensure that validation cloud range is always the same across experiments
        if self.split == "train":
            self.cloud_range = cloud_range
        if self.split == "validate":
            self.cloud_range = [0.01,1.0]

        # get image tif paths, a catalog of used tifs, cloud paths, and a catalog of used cloud masks using appropriate methods
        self.tif_paths, self.tif_catalog = self._get_tif_paths()
        self.cloud_paths, self.cloud_catalog = self._get_cloud_paths()

        self.n_cloudpaths = len(self.cloud_paths)

        self.mean = np.array(mean * 3)[:, np.newaxis, np.newaxis]  # corresponding mean per band for normalization purpose
        self.std = np.array(std * 3)[:, np.newaxis, np.newaxis]  # corresponding std per band for normalization purpose

    def _get_tif_paths(self):
        """
    Retrieve paths to valid image data files and their corresponding metadata catalog.

    This method reads a CSV file containing chip metadata and filters it based on the split, bad pixel percentage,
    and NA count criteria. It then creates a subset of the catalog and extracts valid chip IDs.
    The method constructs paths to the valid image data files and sorts the catalog by chip ID.

    Returns:
        tuple: A tuple containing two elements:
            - truelist (list): A list of pathlib.Path objects representing paths to valid image data files.
            - sorted_catalog (pd.DataFrame): A pandas DataFrame containing sorted metadata of valid chips.

    Note:
        The CSV file should be named "final_chip_tracker.csv" and located within the root directory.
    """
        
        csv = pd.read_csv(self.root_dir.joinpath("final_chip_tracker.csv")) # access chip tracker
        
        # filter csv by split, bad_pct_max and na_count
        catalog = csv.loc[(csv["usage"] == self.split) & (csv["bad_pct_max"] < 5) & (csv["na_count"] == 0)] 
        
        # ensure that validation set is always the same across experiments
        if self.split == "train":
            catalog_subset = catalog.sample(n=self.training_length) # get random sample of the catalog defined by the training length
        else:
            catalog_subset = catalog
        
        itemlist = sorted(catalog_subset["chip_id"].tolist())
        pathlist = [self.image_dir.joinpath(f"{item}_merged.tif") for item in itemlist] # get paths for each item of filtered catalog
        chipslist = list(self.image_dir.glob("*.tif")) # get paths for each item in the image directory
        truelist = sorted(list(set(pathlist) & set(chipslist))) # get only paths from the catalog which represent valid paths in the directory
        sorted_catalog = catalog_subset.sort_values(by="chip_id", ascending=True) # ensure that the catalog is sorted identically to the path list

        return truelist, sorted_catalog

    def _get_cloud_paths(self):
        """
    Retrieve paths to valid cloud mask data files and their corresponding metadata catalog.

    This method reads a CSV file containing cloud mask metadata and filters it based on the split and
    cloud percentage range. It then creates a catalog subset and extracts
    valid cloud mask filenames. The method constructs paths to the valid cloud mask data files.

    Returns:
        tuple: A tuple containing two elements:
            - truelist (list): A list of pathlib.Path objects representing paths to valid cloud mask data files.
            - catalog (pd.DataFrame): A pandas DataFrame containing metadata of valid cloud mask files.

    Note:
        The CSV file should be named "fmask_tracker_balanced.csv" and located within the root directory.
    """
        csv = pd.read_csv(self.root_dir.joinpath("fmask_tracker_balanced.csv")) # access cloud tracker

        # filter csv by usage and cloud cover range defined when initializing the dataset
        catalog = csv.loc[(csv["usage"] == self.split) & (csv["cloud_pct"] <= self.cloud_range[1]) & (csv["cloud_pct"] >= self.cloud_range[0])]
        
        itemlist = sorted(catalog["fmask_name"].tolist()) 
        chipslist = list(self.cloud_dir.glob("*.tif")) # get paths for each item in the cloud directory
        pathlist = [self.cloud_dir.joinpath(f"{item}") for item in itemlist] # get paths for each item of filtered catalog
        truelist = sorted(list(set(pathlist) & set(chipslist))) # get only paths from the catalog which represent valid paths in the directory

        return truelist, catalog
    
    def __len__(self):
        return len(self.tif_paths)

    def __getitem__(self, index):
        """
    Retrieve a combined data sample containing ground truth and cloud mask information.

    This method reads and processes image and cloud mask data for a given index. It loads the merged
    tif file as ground truth and optionally normalizes it. It then creates an empty cloud mask array
    and populates it with cloud scenes based on the mask position and dataset split. The ground truth
    and cloud mask data are combined along new dimensions to create a tensor with the required structure.

    Args:
        index (int): Index of the data sample to retrieve.

    Returns:
        np.ndarray: A numpy array containing the combined data with dimensions (mask-or-image, bands, time steps, height, width).

    Note:
        The method assumes that cloud mask data paths and ground truth data paths have been pre-loaded using the `get_cloud_paths` and `get_tif_paths` methods.
        Additionally, the cloud mask data is read randomly from available paths during training and cyclically during validation.
    """
        def read_tif_as_np_array(path):
            with rasterio.open(path) as src:
                    return src.read()

        # read in merged tif as ground truth
        groundtruth = read_tif_as_np_array(self.tif_paths[index]) # need to normalize here

        if self.normalize:
            groundtruth = np.where(groundtruth == -9999, 0.0001,
                                    (groundtruth - self.mean) / self.std)  # don't normalize on nodata
        else:
            groundtruth = groundtruth * 0.0001  # if not normalize, just rescale
        
        # transpose to bands, time steps, height, width
        groundtruth = groundtruth.reshape(self.num_frames, self.bands, self.img_size, self.img_size)

        # initialize empty cloud mask with same dimensions as ground truth
        cloudbrick = np.zeros_like(groundtruth)

        mask_position = self.mask_position[index % 7] # this loops through the possible combinations of mask position

        # for every specified mask position in training, read in a random cloud scene and add to the block of cloud masks
        if self.split == "train":
            for p in mask_position:
                cloudscene = read_tif_as_np_array(self.cloud_paths[np.random.randint(0,self.n_cloudpaths-1)])
                cloudscene = np.expand_dims(cloudscene, 0)
                cloudbrick[p-1,:,:,:] = cloudscene
                del cloudscene

        if self.split == "validate":
            for p in mask_position:
                # when validating, we remove randomness by looping through the index of the cloud path list
                cloudscene = read_tif_as_np_array(self.cloud_paths[(index + (p-1)) % self.n_cloudpaths]) 
                cloudscene = np.expand_dims(cloudscene, 0)
                cloudbrick[p-1,:,:,:] = cloudscene 
                del cloudscene

        cloudbrick = np.expand_dims(cloudbrick, 0) # adds a dimension at index 0
        groundtruth = np.expand_dims(groundtruth, 0) 
        # concatenate the tensors along the new dimension
        combined_data = np.concatenate((groundtruth, cloudbrick), axis=0).astype(np.float32).transpose(0,2,1,3,4)

        # return tensor with dimensions (mask-or-image, bands, time steps, height, width)
        return combined_data


def visualize_tcc(vis_path, n_epoch, idx, input, input_mask, predicted):
    """
    Generate and save visualizations of inputs and outputs to the model as true color composites.
    This function will only create visualizations for the first tensor in the batch.
    This function creates visualizations of input, predicted, and ground truth images at all time steps.
    The resulting images are saved to the specified visualization path.

    Args:
        vis_path (str): Path to the directory where visualization images will be saved.
        n_epoch (int): Current epoch number.
        idx (int): Index of the sample in the dataset.
        input (torch.Tensor): Input image tensor (shape: [batch_size, bands, time steps, height, width]).
        input_mask (torch.Tensor): Binary mask for input images: 1 is masked while 0 is not masked (shape: [batch_size, bands, time steps, height, width]).
        predicted (torch.Tensor): Predicted image tensor (shape: [batch_size, bands, time steps, height, width]).
    """
    n_timesteps = input.size()[2] # get number of time steps from input

    # initialize empty lists for each category of visualization
    input_list = []
    reconstruction_list = []
    groundtruth_list = []

    # visualize input images and masks for each time step
    for t in range(1, n_timesteps+1):
        input_img = input[0,0:3,t-1,:,:].clone().flip(0) * 3 # get input scenes as [B,H,W], flip bands to RGB and scale by 3 for brightness
        input_mask_img = input_mask[0,0:3,t-1,:,:].clone() # get input mask
        input_masked = torch.where(input_mask_img == 1, input_mask_img, input_img) # make masked areas white by setting all bands to 1
        input_masked = torch.nn.functional.pad(input_masked, (2,2,2,2), value=0) # add 2 pixel wide black border to image
        input_list.append(input_masked) # append to list of input masks, in format [Bands, Height, Width] with B in order R, G, B
        
    # visualize generated model reconstructions for each time step
    for t in range(1, n_timesteps+1):
        input_mask_img = input_mask[0,0:3,t-1,:,:].clone() # get input mask as [B,H,W]
        predicted_img = predicted[0,0:3,t-1,:,:].clone().flip(0) * 3 # get prediceted scenes as [B,H,W], flip bands to RGB and scale by 3 for brightness
        input_img = input[0,0:3,t-1,:,:].clone().flip(0) * 3 # get input scenes, flip to RGB and scale by 3 for brightness
        predicted_unmasked = predicted_img * input_mask_img # get an image of predicted values at masked pixels (mask=1)
        input_masked = input_img * (1-input_mask_img) # get an image of input values at non-masked pixels (mask=0)
        reconstructed_img = predicted_unmasked + input_masked # create composite of input and predicted values
        reconstructed_img = torch.nn.functional.pad(reconstructed_img, (2,2,2,2), value=0) # add 2 pixel wide black border to image
        reconstruction_list.append(reconstructed_img) # append to list of reconstructed images in format [Bands, Height, Width] with B in order R, G, B

    for t in range(1, n_timesteps+1):
        input_img = input[0,0:3,t-1,:,:].clone().flip(0) * 3 # get input scenes, flip to RGB and scale by 3 for brightness
        input_img = torch.nn.functional.pad(input_img, (2,2,2,2), value=0) # add 2 pixel wide black border to image
        groundtruth_list.append(input_img) # append to list of ground truth images in format [Bands, Height, Width] with B in order R, G, B

    # concatenate all images in the lists vertically
    input_list = torch.cat(input_list, dim=1)
    reconstruction_list = torch.cat(reconstruction_list, dim=1)
    groundtruth_list = torch.cat(groundtruth_list, dim=1)

    # concatenate all lists horizontally, creating a single tensor with all visualizations
    full_vis = torch.cat([input_list]+[reconstruction_list]+[groundtruth_list], dim=2)

    # save full visualization tensor to visualization directory, formatted as epoch{n_epoch}_idx{idx}_gen.jpg
    torchvision.utils.save_image(
        full_vis, os.path.join(vis_path, "epoch{:04}_idx{:04}_gen.jpg".format(n_epoch, idx)),
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
    parser.add_argument('--checkpoint', default='/workspace/gfm-gap-filling/pretraining/Prithvi_100M.pt', type=str,
                        help='Path to a checkpoint file to load from.')

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


def train(
        model,
        mask_ratio,
        local_rank,
        rank,
        train_loader,
        optimizer,
        epoch,
        sampler=None,
        profiler=None,
        scheduler=None,
        vis_path=None,
):
    model.train()

    # initialize metrics and send to local rank
    ddp_loss = torch.zeros(2).to(local_rank)
    mask_ratio = torch.zeros(2).to(local_rank)
    ssim = torch.zeros(2).to(local_rank)
    mse = torch.zeros(2).to(local_rank)
    mae = torch.zeros(2).to(local_rank)
    
    # mean and std for normalization purposes
    mean = torch.tensor([495.7316,  814.1386,  924.5740, 2962.5623, 2640.8833, 1740.3031])[None,:,None,None,None].to(local_rank)
    std = torch.tensor([286.9569, 359.3304, 576.3471, 892.2656, 945.9432, 916.1625])[None,:,None,None,None].to(local_rank)

    # initialize torchmetrics classes
    StructuralSimilarity = StructuralSimilarityIndexMeasure(data_range=1.0).to(local_rank)
    mean_squared_error = MeanSquaredError().to(local_rank)
    mean_abs_error = MeanAbsoluteError().to(local_rank)
    
    inner_pbar = tqdm.tqdm(
        range(len(train_loader)), colour="blue", desc="Training Epoch", leave=True
    )

    for i, batch in enumerate(train_loader):

        # get mask batches from dataset
        label_mask_batch = batch[:,1,:,:,:,:].to(local_rank)

        # get input image batches from dataset
        batch = batch[:,0,:,:,:,:].to(local_rank)

        optimizer.zero_grad()

        # run model on mask and input batches
        loss, pred, mask = model(batch, label_mask_batch, mask_ratio)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

        # add mean of mask to running total
        mask_ratio[0] += torch.mean(mask)
        mask_ratio[1] += 1

        # add loss to running total - this is based on z-normalized data
        ddp_loss[0] += loss.item()
        ddp_loss[1] += 1

        # un-normalize the z-normalized input and predicted batch, then re-normalize by dividing by a scaling factor of 0.0001
        # this is to make metrics comparable with metrics from the CGAN baseline, which using scaling factor normalization
        input = (batch.detach() * std + mean) * 0.0001
        input_mask = label_mask_batch.detach() 
        predicted = (model.unpatchify(pred).detach() * std + mean) * 0.0001

        input_masked = input * input_mask # get only input pixels in masked areas
        predicted_masked = predicted * input_mask # get only predicted pixels in masked areas
        
        # get ssim between the masked ground truth and the masked predicted image, only in the center time step
        # this assumes that the only mask is in the central time step, this must be changed for masking at multiple time steps
        ssim_score = StructuralSimilarity(predicted_masked.view(16, -1, 224, 224), input_masked.view(16, -1, 224, 224))

        # add ssim to running total
        ssim[0] += ssim_score.item()
        ssim[1] += 1

        # get mean squared error between the masked ground truth and the masked predicted image, only in the center time step
        # this assumes that the only mask is in the central time step, this must be changed for masking at multiple time steps
        # then, divide by the mean of the input mask in the center time step to normalize the mse to reflect that we are only looking at masked pixels
        mse_score = mean_squared_error(predicted_masked, input_masked)
        mse_score /= (torch.mean(input_mask))
        
        # add mse to running total
        mse[0] += mse_score.item()
        mse[1] += 1

        # get mean absolute error between the masked ground truth and the masked predicted image, only in the center time step
        # this assumes that the only mask is in the central time step, this must be changed for masking at multiple time steps
        # then, divide by the mean of the input mask in the center time step to normalize the mse to reflect that we are only looking at masked pixels
        mae_score = mean_abs_error(predicted_masked, input_masked)
        mae_score /= (torch.mean(input_mask))
        
        # add mae to running total
        mae[0] += mae_score.item()
        mae[1] += 1

        inner_pbar.update(1)
        if profiler:
            profiler.step()
   
        scheduler.step()

    # divide all running metrics to get overall metrics
    train_loss = ddp_loss[0] / ddp_loss[1]
    epoch_mask_ratio = mask_ratio[0] / mask_ratio[1]
    epoch_ssim = ssim[0] / ssim[1]
    epoch_mse = mse[0] / mse[1]
    epoch_mae = mae[0] / mae [1]

    inner_pbar.close()

    # print and return metrics
    print(f"Train Epoch: \t{epoch}, Loss: \t{train_loss:.4f}, Mask Ratio: \t{epoch_mask_ratio:.4f}, SSIM: {epoch_ssim:.4f}, MSE: {epoch_mse:.4f}, MAE: {epoch_mae:.4f}")
    return train_loss, epoch_mask_ratio, epoch_ssim, epoch_mse, epoch_mae


def validation(model, mask_ratio, local_rank, rank, test_loader, n_epoch, vis_path):
    
    model.eval()
    
    # initialize metrics and send to local rank
    ddp_loss = torch.zeros(2).to(local_rank)
    mask_ratio = torch.zeros(2).to(local_rank)
    ssim = torch.zeros(2).to(local_rank)
    mse = torch.zeros(2).to(local_rank)
    mae = torch.zeros(2).to(local_rank)

    # mean and std for normalization purposes
    mean = torch.tensor([495.7316,  814.1386,  924.5740, 2962.5623, 2640.8833, 1740.3031])[None,:,None,None,None].to(local_rank)
    std = torch.tensor([286.9569, 359.3304, 576.3471, 892.2656, 945.9432, 916.1625])[None,:,None,None,None].to(local_rank)

    inner_pbar = tqdm.tqdm(
        range(len(test_loader)), colour="green", desc="Validation Epoch", leave=True
    )

    # initialize torchmetrics classes
    StructuralSimilarity = StructuralSimilarityIndexMeasure(data_range=1.0).to(local_rank)
    mean_squared_error = MeanSquaredError().to(local_rank)
    mean_abs_error = MeanAbsoluteError().to(local_rank)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            
            # get mask batches from dataset
            label_mask_batch = batch[:,1,:,:,:,:].to(local_rank)

            # get input image batches from dataset
            batch = batch[:,0,:,:,:,:].to(local_rank)

            # run model on mask and input batches
            loss, pred, mask = model(batch, label_mask_batch, mask_ratio)
            
            # add mean of mask to running total, adjust to only one mask position
            mask_ratio[0] += torch.mean(mask)
            mask_ratio[1] += 1

            # add loss to running total - this is based on z-normalized data
            ddp_loss[0] += loss.item()
            ddp_loss[1] += 1

            # un-normalize the z-normalized input and predicted batch, then re-normalize by dividing by a scaling factor of 0.0001
            # this is to make metrics comparable with metrics from the CGAN baseline, which using scaling factor normalization
            input = (batch.detach() * std + mean) * 0.0001
            input_mask = label_mask_batch.detach() 
            predicted = (model.unpatchify(pred).detach() * std + mean) * 0.0001

            input_masked = input * input_mask # get only input pixels in masked areas
            predicted_masked = predicted * input_mask # get only predicted pixels in masked areas
            
            # get ssim between the masked ground truth and the masked predicted image, only in the center time step
            # this assumes that the only mask is in the central time step, this must be changed for masking at multiple time steps
            ssim_score = StructuralSimilarity(predicted_masked.view(16, -1, 224, 224), input_masked.view(16, -1, 224, 224))

            # Add ssim to running total
            ssim[0] += ssim_score.item()
            ssim[1] += 1

            # get mean squared error between the masked ground truth and the masked predicted image, only in the center time step
            # this assumes that the only mask is in the central time step, this must be changed for masking at multiple time steps
            # then, divide by the mean of the input mask in the center time step to normalize the mse to reflect that we are only looking at masked pixels
            mse_score = mean_squared_error(predicted_masked, input_masked)
            mse_score /= (torch.mean(input_mask))
            
            # add mse to running total
            mse[0] += mse_score.item()
            mse[1] += 1

            # get mean absolute error between the masked ground truth and the masked predicted image, only in the center time step
            # this assumes that the only mask is in the central time step, this must be changed for masking at multiple time steps
            # then, divide by the mean of the input mask in the center time step to normalize the mse to reflect that we are only looking at masked pixels
            mae_score = mean_abs_error(predicted_masked, input_masked)
            mae_score /= (torch.mean(input_mask))
            
            # add mae to running total
            mae[0] += mae_score.item()
            mae[1] += 1
            
            inner_pbar.update(1)

    # divide all running metrics to get overall metrics
    val_loss = ddp_loss[0] / ddp_loss[1]
    epoch_mask_ratio = mask_ratio[0] / mask_ratio[1]
    epoch_ssim = ssim[0] / ssim[1]
    epoch_mse = mse[0] / mse[1]
    epoch_mae = mae[0] / mae [1]
    
    inner_pbar.close()
    
    # print and return metrics
    print(f"Validation Loss: {val_loss:.4f}, Mask Ratio: \t{epoch_mask_ratio:.4f}, SSIM: {epoch_ssim:.4f}, MSE: {epoch_mse:.4f}, MAE: {epoch_mae:.4f}")
    return val_loss, epoch_mask_ratio, epoch_ssim, epoch_mse, epoch_mae


def fsdp_main(args):
    """main process, run within each individual GPU process"""

    start_time = f"{time.strftime('%Y-%m-%d %H:%M:%S')}"

    # debug nan gradient
    torch.autograd.set_detect_anomaly(True)

    # these are configured to run the code with python rather than torchrun in order to run parallel experiments
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
    num_workers = args.data_loader_num_workers

    # model related
    num_layers = args.num_layers
    patch_size = args.patch_size
    embed_dim = args.embed_dim
    num_heads = args.num_heads
    mask_ratio = args.mask_ratio
    tubelet_size = args.tubelet_size
    checkpoint = args.checkpoint

    # training related
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.num_epochs

    # logging related
    job_id = f"{training_length}-fair-bs16-{time.strftime('%Y-%m-%d_%H-%M-%S')}"

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
    with open(os.path.join(job_info_dir, f'{job_id}.yaml'), 'w') as f:
        yaml.safe_dump(params_dict, f, default_flow_style=None, sort_keys=False)
    
    # set seed in a way that:
    # 1. ensure reproducibility
    # 2. make sure each gpu has different seed to make sure different
    # gpus crop the same images randomly
    random.seed(2022)
    torch.cuda.manual_seed(2022)
    torch.manual_seed(2022)

    # distributed setup
    # dist.init_process_group("nccl")
    # os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    # os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    # create model - use if not loading from a checkpoint
    # model = MaskedAutoencoderViT(img_size=img_size, patch_size=patch_size,
    #              num_frames=num_frames, tubelet_size=tubelet_size,
    #              in_chans=6, embed_dim=embed_dim, depth=num_layers, num_heads=num_heads,
    #              decoder_embed_dim=int(embed_dim/2), decoder_depth=8, decoder_num_heads=num_heads,
    #              mlp_ratio=4., norm_layer=functools.partial(torch.nn.LayerNorm, eps=1e-6), norm_pix_loss=False)

    def prepare_model(checkpoint, arch='mae_vit_base_patch16'):
        # build model
        model = getattr(mae.models_mae, arch)()
        # load model
        checkpoint_file = torch.load(checkpoint, map_location=f'cuda:{local_rank}')
        msg = model.load_state_dict(checkpoint_file, strict=False)
        print(msg)
        return model

    # load pretrained weights along with the version of the model that matches them
    model = prepare_model(checkpoint, 'mae_vit_base_patch16')
    print('Model loaded.')

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> model has {total_params / 1e6} Million params.\n")

    # create training dataset
    train_dataset = CombinedDataset(train_dir, split="train", num_frames=3, img_size=224, bands=6, cloud_range=cloud_range, normalize=True, training_length=training_length)
    if rank == 0:
        print(f"--> Training set len = {len(train_dataset)}")
    if rank == 0:
        print(f"--> Training set masks = {train_dataset.n_cloudpaths}")

    # create validation dataset - note that cloud range is constant to ensure that validation metrics are comparable across experiments
    val_dataset = CombinedDataset(train_dir, split="validate", num_frames=3, img_size=224, bands=6, cloud_range=[0.01,1], normalize=True)
    if rank == 0:
        print(f"--> Validation set len = {len(val_dataset)}")
    if rank == 0:
        print(f"--> Validation set masks = {val_dataset.n_cloudpaths}")
    
    # save catalogs for all image and cloud chips to the csv log directory
    train_dataset.cloud_catalog.to_csv(os.path.join(csv_log_dir, "train_cloud_catalog.csv"), index=False)
    val_dataset.cloud_catalog.to_csv(os.path.join(csv_log_dir, "val_cloud_catalog.csv"), index=False)
    train_dataset.tif_catalog.to_csv(os.path.join(csv_log_dir, "train_tif_catalog.csv"), index=False)
    val_dataset.tif_catalog.to_csv(os.path.join(csv_log_dir, "val_tif_catalog.csv"), index=False)

    # initialize samplers, note that we use a sequential sampler for identical validation sets each time
    train_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    # get arguments for training and testing loaders and initialize them
    train_kwargs = {"batch_size": batch_size, "sampler": train_sampler}
    test_kwargs = {"batch_size": batch_size, "sampler": val_sampler}
    common_kwargs = {
        "pin_memory": False,
        "drop_last": True
    }
    train_kwargs.update(common_kwargs)
    test_kwargs.update(common_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)

    torch.cuda.set_device(local_rank)

    torch.cuda.empty_cache()

    model = model.to(torch.cuda.current_device())

    # optimizer and learning rate decay
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = OneCycleLR(optimizer, max_lr=lr*10, steps_per_epoch=len(train_loader), epochs=epochs)

    # initialize best validation loss and best validation ssim
    best_val_loss = float("inf")
    best_val_ssim = 0

    # --- main training loop
    if rank == 0:
        dur = []
        train_acc_tracking = []
        val_acc_tracking = []
        dq = deque(maxlen=3)
        training_start_time = time.time()

    # torch profiler
    torch_profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "profile_traces"
        ),
        profile_memory=True,
        with_stack=False,
        record_shapes=True,
    )
    torch_profiler = None

    # log writers
    if rank == 0:
        tensorboard_writer = SummaryWriter(tensorboard_log_dir)
        os.makedirs(csv_log_dir, exist_ok=True)
        log_writer = open(os.path.join(csv_log_dir, "summary.txt"), "a")

    if rank == 0:
        mem_alloc_tracker = []

    # -- Start Training -----
    for epoch in range(1, epochs + 1):
        print('training ' + str(epoch) + ' rank ' + str(rank))
        
        if rank == 0:
            print(f"\n--> Starting Epoch {epoch}")

            t0 = time.time()
        
        # run training for epoch
        train_loss, train_mask_ratio, train_ssim, train_mse, train_mae = train(
            model,
            mask_ratio,
            local_rank,
            rank,
            train_loader,
            optimizer,
            epoch,
            sampler=train_sampler,
            profiler=torch_profiler,
            scheduler=scheduler,
            vis_path=vis_dir,
        )

        # run validation for epoch
        curr_val_loss, val_mask_ratio, val_ssim, val_mse, val_mae = validation(model, mask_ratio, local_rank, rank, test_loader, epoch, vis_path=vis_dir)

        # write logs in two formats: tensorboard and csv.
        if rank == 0:
            tensorboard_writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
            tensorboard_writer.add_scalars("Loss", {
                "train": train_loss,
                "test":  curr_val_loss
            }, epoch)
            log_writer.write(f"{epoch},{scheduler.get_last_lr()[0]},{train_loss},{train_mask_ratio},{train_ssim},{train_mse},{train_mae},{curr_val_loss},{val_mask_ratio},{val_ssim},{val_mse},{val_mae}\n")
            # flush on each write to avoid log loss due to unexpected exit
            tensorboard_writer.flush()
            log_writer.flush()

        # saving and statistics
        if rank == 0:
            print(f"--> epoch {epoch} completed...entering save and stats zone")

            dur.append(time.time() - t0)
            train_acc_tracking.append(train_loss.item())

            val_acc_tracking.append(curr_val_loss.item())

            mem_alloc_tracker.append(
                round((torch.cuda.memory_allocated() / 1024 ** 3), ndigits=4)
            )

        # save this epochs checkpoint if val loss is current best
        if curr_val_loss < best_val_loss:
            if rank == 0:
                print(f"--> saving model ...")
                filename = "model_best.pt"
                checkpoint_file = os.path.join(ckpt_dir, filename)
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_file)
                print(f"--> saved {checkpoint_file} to COS")

        # save this epochs checkpoint if val ssim is current best
        if val_ssim > best_val_ssim:
            if rank == 0:
                print(f"--> saving model ...")
                filename = "model_best_ssim.pt"
                checkpoint_file = os.path.join(ckpt_dir, filename)
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_file)
                print(f"--> saved {checkpoint_file} to COS")

        # announce new val loss record:
        if rank == 0 and curr_val_loss < best_val_loss:
            best_val_loss = curr_val_loss
            print(f"-->>>> New Val Loss Record: {best_val_loss}")

        if rank == 0 and val_ssim > best_val_ssim:
            best_val_ssim = val_ssim
            print(f"-->>>> New Val Loss Record: {best_val_ssim}")

    if rank == 0:

        total_training_time = time.time() - training_start_time
        print(f"Total training time = {total_training_time:.2f}")
        print("Times per epoch:")
        for i, val in enumerate(dur):
            print(f"epoch {i}, time {val:.2f}")
        print()

        # training is done...show some training stats for memory use.
        print(f"total memory allocated: {mem_alloc_tracker}")

        print(f"Training accuracy: {train_acc_tracking}")
        print(f"Validation accuracy: {val_acc_tracking}")
        print(f"\n Best Val accuracy: {best_val_loss}")

        # memory summary
        print(f"CUDA Memory Summary After Last training:\n {torch.cuda.memory_summary()}")

        # close tensorboard writer
        tensorboard_writer.flush()
        tensorboard_writer.close()
        log_writer.close()

        # update job info file
        with open(os.path.join(job_info_dir, f'{job_id}.yaml'), 'r') as f:
            params_dict = yaml.safe_load(f)
            params_dict['job_finish_time'] = f"{time.strftime('%Y-%m-%d %H:%M:%S')}"

        with open(os.path.join(job_info_dir, f'{job_id}.yaml'), 'w') as f:
            yaml.safe_dump(params_dict, f, default_flow_style=None, sort_keys=False)

# ------------------ Main functions above ------------

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    fsdp_main(args)

