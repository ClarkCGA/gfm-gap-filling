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
                 normalize=True,
                 # small trick: as all original values are integers, we make mean values as non-integer
                 # so normalized values have no 0, so that we can mark nodata as 0 because it is a good practice
                 # to fill nodata as mean/median.
                 mean=[431.5, 713.5, 747.5, 2512.5], std=[336, 388, 521, 1062], indices=None):
        self.root_dir = pathlib.Path(data_path)
        self.image_dir = self.root_dir.joinpath("chips_filtered/")
        self.cloud_dir = self.root_dir.joinpath("cloud_mask/")
        self.split = split
        self.num_frames = num_frames
        self.mask_position = [2]
        self.img_size = img_size
        self.bands = bands
        self.num_hls_bands = num_hls_bands
        self.cloud_range = cloud_range
        self.normalize = normalize

        self.tif_paths = self._get_tif_paths()
        self.cloud_paths, self.cloud_catalog = self._get_cloud_paths()

        self.n_cloudpaths = len(self.cloud_paths)
        self.mean = mean  # corresponding mean per band for normalization purpose
        self.std = std  # corresponding std per band for normalization purpose

    # Create list of all merged image chips
    def _get_tif_paths(self):
        csv = pd.read_csv(self.root_dir.joinpath("final_chip_tracker.csv"))
        catalog = csv.loc[(csv["usage"] == self.split) & (csv["bad_pct_max"] < 5) & (csv["na_count"] == 0)]
        itemlist = sorted(catalog["chip_id"].tolist())
        pathlist = [self.image_dir.joinpath(f"{item}_merged.tif") for item in itemlist]
        chipslist = list(self.image_dir.glob("*.tif"))
        truelist = sorted(list(set(pathlist) & set(chipslist)))
        return truelist

    # Create list of all paths to clouds
    def _get_cloud_paths(self):
        csv = pd.read_csv(self.root_dir.joinpath("fmask_tracker.csv"))
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

def visualize_tcc(vis_path, n_epoch, idx, input, input_mask, processed_mask, predicted):

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
        processed_mask_img = processed_mask[0,0:3,t-1,:,:].clone()
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
     #   mask_train_loader_list,
        optimizer,
        epoch,
        sampler=None,
        profiler=None,
        scheduler=None,
        vis_path=None,
):
    model.train()
    ddp_loss = torch.zeros(2).to(local_rank)
    mask_ratio = torch.zeros(2).to(local_rank)
    ssim = torch.zeros(2).to('cpu')
    mse = torch.zeros(2).to('cpu')

    StructuralSimilarity = StructuralSimilarityIndexMeasure(data_range=1.0)
    mean_squared_error = MeanSquaredError()
    # if sampler:
    #     sampler.set_epoch(epoch)
    inner_pbar = tqdm.tqdm(
        range(len(train_loader)), colour="blue", desc="Training Epoch", leave=True
    )
    # start = time.time()
    for i, batch in enumerate(train_loader):

        label_mask_batch = batch[:,1,:,:,:,:]

        batch = batch[:,0,:,:,:,:]


        
        optimizer.zero_grad()

       # label_mask_batch = mask_train_loader_list[i]

        loss, pred, mask = model(batch.to(local_rank), label_mask_batch.to(local_rank), mask_ratio)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

        mask_ratio[0] += torch.mean(mask) * 3
        mask_ratio[1] += 1

        ddp_loss[0] += loss.item()
        ddp_loss[1] += 1

        input = batch.detach().cpu()
        input_mask = label_mask_batch.detach().cpu()
        processed_mask = model.unpatchify(mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1])).detach().cpu()
        predicted = model.unpatchify(pred).detach().cpu()
        input_masked = input * processed_mask
        predicted_masked = predicted * processed_mask
        
        ssim_score = StructuralSimilarity(predicted_masked[:,:,1,:,:], input_masked[:,:,1,:,:])

        ssim[0] += ssim_score
        ssim[1] += 1

        mse_score = mean_squared_error(predicted_masked[:,:,1,:,:], input_masked[:,:,1,:,:])
        mse_score /= (torch.mean(mask).detach().cpu() * 3)
        
        mse[0] += mse_score
        mse[1] += 1

        inner_pbar.update(1)
        if profiler:
            profiler.step()
   
        scheduler.step()


    # consolidate final loss number - do not use .reduce here, requires global synch
    # dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    train_loss = ddp_loss[0] / ddp_loss[1]
    epoch_mask_ratio = mask_ratio[0] / mask_ratio[1]
    epoch_ssim = ssim[0] / ssim[1]
    epoch_mse = mse[0] / mse[1]
    inner_pbar.close()

    print(f"Train Epoch: \t{epoch}, Loss: \t{train_loss:.4f}, Mask Ratio: \t{epoch_mask_ratio:.4f}, SSIM: {epoch_ssim:.4f}, MSE: {epoch_mse:.4f}")
    return train_loss, epoch_mask_ratio, epoch_ssim, epoch_mse


def validation(model, mask_ratio, local_rank, rank, test_loader, n_epoch, vis_path):
    model.eval()
    ddp_loss = torch.zeros(2).to(local_rank)
    mask_ratio = torch.zeros(2).to(local_rank)
    ssim = torch.zeros(2).to('cpu')
    mse = torch.zeros(2).to('cpu')
    inner_pbar = tqdm.tqdm(
        range(len(test_loader)), colour="green", desc="Validation Epoch", leave=True
    )
    
    StructuralSimilarity = StructuralSimilarityIndexMeasure(data_range=1.0)
    mean_squared_error = MeanSquaredError()

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
            
            input = batch.detach().cpu()
            input_mask = label_mask_batch.detach().cpu()
            processed_mask = model.unpatchify(mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1])).detach().cpu()
            predicted = model.unpatchify(pred).detach().cpu()
            input_masked = input * processed_mask
            predicted_masked = predicted * processed_mask
            
            ssim_score = StructuralSimilarity(predicted_masked[:,:,1,:,:].detach().cpu(), input_masked[:,:,1,:,:].detach().cpu())

            ssim[0] += ssim_score
            ssim[1] += 1
  
            mse_score = mean_squared_error(predicted_masked[:,:,1,:,:].detach().cpu(), input_masked[:,:,1,:,:].detach().cpu())
            mse_score /= (torch.mean(mask).detach().cpu() * 3)
            
            mse[0] += mse_score
            mse[1] += 1
            
            if i == 8:
                if vis_path is not None:
                    os.makedirs(vis_path, exist_ok=True)
                    visualize_tcc(vis_path, n_epoch, i, input, input_mask, processed_mask, predicted)

        # dist.barrier()

    # dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    val_loss = ddp_loss[0] / ddp_loss[1]
    epoch_mask_ratio = mask_ratio[0] / mask_ratio[1]
    epoch_ssim = ssim[0] / ssim[1]
    epoch_mse = mse[0] / mse[1]
    
    inner_pbar.close()
    print(f"Validation Loss: {val_loss:.4f}, Mask Ratio: \t{epoch_mask_ratio:.4f}, SSIM: {epoch_ssim:.4f}, MSE: {epoch_mse:.4f}")
    return val_loss, epoch_mask_ratio, epoch_ssim, epoch_mse


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

    # training related
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.num_epochs

    # logging related
    job_id = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}"

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

    # create model
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

    model = prepare_model(checkpoint, 'mae_vit_base_patch16')
    print('Model loaded.')

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> model has {total_params / 1e6} Million params.\n")

    # ____________ create batch dataset
    train_dataset = CombinedDataset(train_dir, split="train", num_frames=3, img_size=224, bands=6, cloud_range=cloud_range,
                              # random_cropping=random_cropping, remove_cloud=True, 
                               normalize=False)
    if rank == 0:
        print(f"--> Training set len = {len(train_dataset)}")
    if rank == 0:
        print(f"--> Training set masks = {train_dataset.n_cloudpaths}")

    val_dataset = CombinedDataset(train_dir, split="validate", num_frames=3, img_size=224, bands=6, cloud_range=cloud_range,
                              # random_cropping=random_cropping, remove_cloud=True, 
                               normalize=False)
    if rank == 0:
        print(f"--> Validation set len = {len(val_dataset)}")
    if rank == 0:
        print(f"--> Validation set masks = {val_dataset.n_cloudpaths}")
    
    train_dataset.cloud_catalog.to_csv(os.path.join(job_info_dir, "train_cloud_catalog.csv"), index=False)
    val_dataset.cloud_catalog.to_csv(os.path.join(job_info_dir, "val_cloud_catalog.csv"), index=False)

    train_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_kwargs = {"batch_size": batch_size, "sampler": train_sampler}
    test_kwargs = {"batch_size": batch_size, "sampler": val_sampler}
    common_kwargs = {
        #"num_workers": num_workers,
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
    # model = DDP(model, device_ids=[torch.cuda.current_device()])

    # optimizer and learning rate decay
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = OneCycleLR(optimizer, max_lr=lr*10, steps_per_epoch=len(train_loader), epochs=epochs)

    best_val_loss = float("inf")

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

    # Log Writers
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
        train_loss, train_mask_ratio, train_ssim, train_mse = train(
            model,
            mask_ratio,
            local_rank,
            rank,
            train_loader,
          #  mask_train_loader_list,
            optimizer,
            epoch,
            sampler=train_sampler,
            profiler=torch_profiler,
            scheduler=scheduler,
            vis_path=vis_dir,
        )

        curr_val_loss, val_mask_ratio, val_ssim, val_mse = validation(model, mask_ratio, local_rank, rank, test_loader, epoch, vis_path=vis_dir)

        # Write logs in two formats: tensorboard and csv.
        if rank == 0:
            tensorboard_writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
            tensorboard_writer.add_scalars("Loss", {
                "train": train_loss,
                "test":  curr_val_loss
            }, epoch)
            log_writer.write(f"{epoch},{scheduler.get_last_lr()[0]},{train_loss},{train_mask_ratio},{train_ssim},{curr_val_loss},{train_mse},{val_mask_ratio},{val_ssim},{val_mse}\n")
            # flush on each write to avoid log loss due to unexpected exit
            tensorboard_writer.flush()
            log_writer.flush()

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

        # announce new val loss record:
        if rank == 0 and curr_val_loss < best_val_loss:
            best_val_loss = curr_val_loss
            print(f"-->>>> New Val Loss Record: {best_val_loss}")

    # init_end_event.record()
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

        # Update job info file
        with open(os.path.join(job_info_dir, f'{job_id}.yaml'), 'r') as f:
            params_dict = yaml.safe_load(f)
            params_dict['job_finish_time'] = f"{time.strftime('%Y-%m-%d %H:%M:%S')}"

        with open(os.path.join(job_info_dir, f'{job_id}.yaml'), 'w') as f:
            yaml.safe_dump(params_dict, f, default_flow_style=None, sort_keys=False)

    # all done, set barrier to ensure all GPU's complete, and then cleanup
    # dist.barrier()
    # dist.destroy_process_group()


# ------------------ Main functions above ------------


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    fsdp_main(args)

