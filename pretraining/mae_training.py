import argparse
import functools
import time
from collections import deque
import pathlib

import torch
import torch.distributed as dist
import torch.optim as optim
import tqdm
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import ConcatDataset

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
    def __init__(self, data_path, split="train", num_frames=3, img_size=224, bands=["CDL"], num_hls_bands = 6,
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
        self.num_hls_bands = num_hls_bands,
        self.normalize = normalize

        self.tif_paths = self._get_tif_paths()[:32]
        self.cloud_paths = self._get_cloud_paths()[:32]

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
        truelist = list(set(pathlist) & set(chipslist))
        return truelist

    # Create list of all paths to clouds
    def _get_cloud_paths(self):
        csv = pd.read_csv(self.root_dir.joinpath("fmask_tracker.csv"))
        catalog = csv.loc[(csv["usage"] == self.split) & (csv["cloud_pct"] <= .6) & (csv["cloud_pct"] >= .3)]
        itemlist = sorted(catalog["fmask_name"].tolist())
        chipslist = list(self.cloud_dir.glob("*.tif"))
        pathlist = [self.cloud_dir.joinpath(f"{item}") for item in itemlist]
        truelist = list(set(pathlist) & set(chipslist))
        return pathlist
    
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
            groundtruth = groundtruth * 0.0002  # if not normalize, just rescale
        
        # Transpose to bands, time steps, height, width
        groundtruth = groundtruth.reshape(self.num_frames, self.bands, self.img_size, self.img_size)

        # Initialize empty cloud mask with same dimensions as ground truth
        cloudbrick = np.zeros_like(groundtruth)

        # For every specified mask position, read in a random cloud scene and add to the block of cloud masks
        for p in self.mask_position:
            cloudscene = read_tif_as_np_array(self.cloud_paths[np.random.randint(0,self.n_cloudpaths-1)])
            cloudscene = np.expand_dims(cloudscene, 0)
            cloudbrick[p-1,:,:,:] = cloudscene # Check if this works, the code should assign cloud scene to ALL these values in the 4 channels indexed.
            del cloudscene

        cloudbrick = np.expand_dims(cloudbrick, 0) # Adds a dimension at index 0
        groundtruth = np.expand_dims(groundtruth, 0) 
        # Concatenate the tensors along the new dimension
        combined_data = np.concatenate((groundtruth, cloudbrick), axis=0).astype(np.float32).transpose(0,2,1,3,4)

        # A tensor with dimensions (mask-or-image, bands, time steps, height, width)
        return combined_data
  
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
    parser.add_argument('--checkpoint', default='', type=str,
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

    if sampler:
        sampler.set_epoch(epoch)
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

        ddp_loss[0] += loss.item()
        ddp_loss[1] += 1

        inner_pbar.update(1)
        if profiler:
            profiler.step()
   
        scheduler.step()

        if epoch == 1 and i <= 5:
            if vis_path is not None:
                os.makedirs(vis_path, exist_ok=True)
                torch.save(batch.detach().cpu(), os.path.join(vis_path, f'input_{i}.pt'))
                torch.save(label_mask_batch.detach().cpu(), os.path.join(vis_path, f'input_mask_{i}.pt'))
                torch.save(model.module.unpatchify(mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1])).detach().cpu(), os.path.join(vis_path, f'mask_{i}.pt'))
                torch.save(model.module.unpatchify(pred).detach().cpu(), os.path.join(vis_path, f'pred_{i}.pt'))
        dist.barrier()

    # consolidate final loss number - do not use .reduce here, requires global synch
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    train_loss = ddp_loss[0] / ddp_loss[1]
    inner_pbar.close()

    print(f"Train Epoch: \t{epoch}, Loss: \t{train_loss:.4f}")
    return train_loss


def validation(model,  mask_ratio, local_rank, rank, test_loader):
    model.eval()
    ddp_loss = torch.zeros(2).to(local_rank)
    inner_pbar = tqdm.tqdm(
        range(len(test_loader)), colour="green", desc="Validation Epoch", leave=True
    )
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            label_mask_batch = batch[:,1,:,:,:,:]

            batch = batch[:,0,:,:,:,:]
          #  label_mask_batch = mask_val_loader_list[i]
            loss, pred, mask = model(batch.to(local_rank), label_mask_batch.to(local_rank), mask_ratio)

            ddp_loss[0] += loss.item()  # sum up batch loss
            ddp_loss[1] += 1

            inner_pbar.update(1)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    val_loss = ddp_loss[0] / ddp_loss[1]

    inner_pbar.close()
    print(f"Validation Loss: {val_loss:.4f}")
    return val_loss


def fsdp_main(args):
    """main process, run within each individual GPU process"""

    # TODO: can we get the time the job was submitted?
    start_time = f"{time.strftime('%Y-%m-%d %H:%M:%S')}"

    # debug nan gradient
    torch.autograd.set_detect_anomaly(True)

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = 0
    world_size = int(os.environ["WORLD_SIZE"])

    ### Configs
    # data loader related
    train_dir = args.train_dir
    val_dir = args.val_dir
    mask_dir = args.mask_dir
    num_frames = args.num_frames
    img_size = args.img_size
    bands = args.bands
    num_hls_bands = len(bands)
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
    dist.init_process_group("nccl")
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    # create model
    model = MaskedAutoencoderViT(img_size=img_size, patch_size=patch_size,
                 num_frames=num_frames, tubelet_size=tubelet_size,
                 in_chans=6, embed_dim=embed_dim, depth=num_layers, num_heads=num_heads,
                 decoder_embed_dim=int(embed_dim/2), decoder_depth=8, decoder_num_heads=num_heads,
                 mlp_ratio=4., norm_layer=functools.partial(torch.nn.LayerNorm, eps=1e-6), norm_pix_loss=False)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> model has {total_params / 1e6} Million params.\n")

    # ____________ create batch dataset
    train_dataset = CombinedDataset(train_dir, split="train", num_frames=3, img_size=224, bands=6,
                              # random_cropping=random_cropping, remove_cloud=True, 
                               normalize=False)
    if rank == 0:
        print(f"--> Training set len = {len(train_dataset)}")
    if rank == 0:
        print(f"--> Training set masks = {train_dataset.n_cloudpaths}")

    val_dataset = CombinedDataset(train_dir, split="validate", num_frames=3, img_size=224, bands=6,
                              # random_cropping=random_cropping, remove_cloud=True, 
                               normalize=False)
    if rank == 0:
        print(f"--> Validation set len = {len(val_dataset)}")
    if rank == 0:
        print(f"--> Validation set masks = {val_dataset.n_cloudpaths}")
    
  #  train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
  #  val_sampler = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size, shuffle=True)

  #  print('train_sampler')
 #   print(len(train_sampler))
  #  print(dir(train_sampler))


    train_kwargs = {"batch_size": batch_size, "sampler": train_sampler}
    test_kwargs = {"batch_size": 1, "sampler": val_sampler}
    common_kwargs = {
        #"num_workers": num_workers,
        "pin_memory": False,
        "drop_last": True
    }
    train_kwargs.update(common_kwargs)
    test_kwargs.update(common_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)

 #   mask_train_loader = torch.utils.data.DataLoader(mask_train_dataset, **train_kwargs)
  #  mask_test_loader = torch.utils.data.DataLoader(mask_val_dataset, **test_kwargs)

  #  print('mtd')
  #  mask_train_loader_list = list(mask_train_loader)
  #  mask_val_loader_list = list(mask_test_loader)
  #  print(mask_train_loader_list[0])


    # print('combined train_loader')
    # for i, batch in enumerate(train_loader):
    #  #   print(i)
    #  #   print(batch.shape)
    #     x = batch[:,0,:,:,:,:]
    #   #  print(x)
    #   #  print(type(x))
    #     print(x.shape)
    #     mask = batch[:,1,:,:,:,:]
    #    # print(mask)
    #   #  print(mask.shape)
    #     #print(mask_train_loader_list[i].shape)
 #   print(len(train_loader))
  #  print(type(train_loader))
  #  print(train_loader[0])
    

 #   print('mask loader')
  #  print(len(mask_train_loader))
  #  print(len(mask_train_loader[0]))
    # print('mask_train_dataset')
    # print(len(mask_train_dataset))
    # print(mask_train_dataset[0].shape)

    # print('mask_test dataset')
    # print(len(mask_val_dataset))
    # print(mask_val_dataset[0].shape)
          
    
    torch.cuda.set_device(local_rank)

    torch.cuda.empty_cache()

    model = model.to(torch.cuda.current_device())
    model = DDP(model, device_ids=[torch.cuda.current_device()])

    if checkpoint:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        model.load_state_dict(
            torch.load(checkpoint, map_location=map_location))

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
        train_loss = train(
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

        curr_val_loss = validation(model, mask_ratio, local_rank, rank, test_loader)

        # Write logs in two formats: tensorboard and csv.
        if rank == 0:
            tensorboard_writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
            tensorboard_writer.add_scalars("Loss", {
                "train": train_loss,
                "test":  curr_val_loss
            }, epoch)
            log_writer.write(f"{epoch},{scheduler.get_last_lr()[0]},{train_loss},{curr_val_loss}\n")
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
                filename = f"epoch-{epoch}-loss-{round(curr_val_loss.item(), 4)}.pt"
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

