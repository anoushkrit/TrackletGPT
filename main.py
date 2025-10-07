from tools import pretrain_run_net as pretrain
from tools import finetune_run_net as finetune
from tools import test_run_net as test_net
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
import time
import os
import torch
from tensorboardX import SummaryWriter
from torchstat import stat


def main():
    # args
    args = parser.get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # define the tensorboard writer
    if not args.test:
        if not args.distributed or torch.distributed.get_rank() == 0:
            train_writer = SummaryWriter(
                os.path.join(args.tfboard_path, 'train'))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
        else:
            train_writer = None
            val_writer = None
    # config
    config = get_config(args, logger=logger)
    # batch size
    if args.distributed:
        assert config.total_bs % world_size == 0
        config.dataset.train.others.bs = config.total_bs // world_size
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs // world_size * 2
        config.dataset.val.others.bs = config.total_bs // world_size * 2
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs // world_size
    else:
        config.dataset.train.others.bs = config.total_bs
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs * 2
        # config.dataset.val.others.bs = config.total_bs * 2
        config.dataset.val.others.bs = config.total_bs
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs
    # log
    log_args_to_file(args, 'args', logger=logger)
    log_config_to_file(config, 'config', logger=logger)
    # exit()
    logger.info(f'Distributed training: {args.distributed}')
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        # seed + rank, for augmentation
        misc.set_random_seed(args.seed + args.local_rank,
                             deterministic=args.deterministic)
    if args.distributed:
        global_rank = torch.distributed.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])  # comes from torchrun
        world_size = torch.distributed.get_world_size()

        # Debug: print CUDA_VISIBLE_DEVICES and device count
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        print(f"[Rank {global_rank}] CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
        print(f"[Rank {global_rank}] local_rank: {local_rank}, torch.cuda.device_count(): {torch.cuda.device_count()}")

        if cuda_visible_devices is None:
            print(f"[Rank {global_rank}] WARNING: CUDA_VISIBLE_DEVICES is not set. Device assignment may be incorrect.")

        # Check local_rank validity
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(f"[Rank {global_rank}] local_rank {local_rank} exceeds available CUDA devices ({torch.cuda.device_count()})")

        # set the correct GPU device for this process
        torch.cuda.set_device(local_rank)

        if global_rank == 0:
            print_log(f"[Rank {global_rank}] Master process initialized "
                f"(world size: {world_size})", logger="main")
        else:
            print_log(f"[Rank {global_rank}] Worker process initialized on GPU {local_rank}", logger="main")

    if args.shot != -1:
        config.dataset.train.others.shot = args.shot
        config.dataset.train.others.way = args.way
        config.dataset.train.others.fold = args.fold
        config.dataset.val.others.shot = args.shot
        config.dataset.val.others.way = args.way
        config.dataset.val.others.fold = args.fold

    # run
    if args.test:
        test_net(args, config)
    else:
        if args.finetune_model or args.scratch_model:
            finetune(args, config, train_writer, val_writer)
        else:
            pretrain(args, config, train_writer, val_writer)


if __name__ == '__main__':
    main()
