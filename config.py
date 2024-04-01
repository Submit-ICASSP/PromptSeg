import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Path to dataset info json, could be overwritten by command line argument
_C.DATA.DATASET_INFO_JSON_DIR = ''
# Dataset name
_C.DATA.DATASET = 'RadiologyDataset'
# Sub dataset list in Main dataset
_C.DATA.DATASET_LIST = []
# Sub dataset name in Main dataset for validation
_C.DATA.EVAL_DATASET_LIST = []
# exclusive class name in each Sub dataset
_C.DATA.EXCLUSIVE_CLASS_LIST = []
# Class name in each Sub dataset
_C.DATA.CLASS_DICT_JSON = ''
# Random sample ratio
_C.DATA.RANDOM_SAMPLE_RATIO = 0.1
# Random sample first image in a sentence or not
_C.DATA.RANDOM_FIRST = False
# Reset sentences list every epoch or not
_C.DATA.RESET_SENTENCE = True
# Image channel
_C.DATA.IMAGE_CHANNEL = 3

# Input image size
_C.DATA.IMG_SIZE = 256
# Input image per sequence, sequence length = 2 * pairs * (img_size/patch_size) ** 2 
_C.DATA.PAIRS_NUM_PER_SEQUENCE = 4
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8
# Number of classes in the dataset
_C.DATA.NUM_CLASSES = 2
# Enable task transform
_C.DATA.TASK_TRANSFORM = False
# Task transform type, random means choose a random task from task list, all means use all tasks
_C.DATA.TASK_TRANSFORM_TYPE = 'random'
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'SAMMed+GPT2'
# Model name
_C.MODEL.NAME = 'SAMMed-vit-b+GPT2-tiny'
# Use pretrained model
_C.MODEL.PRETRAINED = ''
# Pretrained weight from checkpoint for encoder, 
# could be SAMMed pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED_ENCODER = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1

# ---------------------------------------------
# PromptSeg model
# ---------------------------------------------

# Image encoder parameters
_C.MODEL.ENCODER = CN()
_C.MODEL.ENCODER.NAME = 'SAMMed-ViT-base'
_C.MODEL.ENCODER.DEPTH = 12
_C.MODEL.ENCODER.HIDDEN_SIZE = 768
_C.MODEL.ENCODER.MLP_RATIO = 4.
_C.MODEL.ENCODER.NUM_HEADS = 12
_C.MODEL.ENCODER.QKV_BIAS = True
_C.MODEL.ENCODER.RELATIVE_POSITION_EMBEDDING = True
_C.MODEL.ENCODER.GLOBAL_ATTENTION_INDEX = [2, 5, 8, 11]
_C.MODEL.ENCODER.WINDOW_SIZE = 14
_C.MODEL.ENCODER.PATCH_SIZE = 16
_C.MODEL.ENCODER.OUT_CHANS = 256
_C.MODEL.ENCODER.ADAPTER = True
_C.MODEL.ENCODER.NORM = 'LayerNorm'
_C.MODEL.ENCODER.NORM_EPS = 1e-6
_C.MODEL.ENCODER.FREEZE = False

# Image decoder parameters
_C.MODEL.DECODER = CN()
_C.MODEL.DECODER.NAME = 'GPT2'
_C.MODEL.DECODER.INPUT_CHANS = 256
_C.MODEL.DECODER.DEPTH = 12
_C.MODEL.DECODER.HIDDEN_SIZE = 768
_C.MODEL.DECODER.FFN_HIDDEN_SIZE = None
_C.MODEL.DECODER.MLP_RATIO = 4.
_C.MODEL.DECODER.NUM_HEADS = 12
_C.MODEL.DECODER.SCALE_ATTENTION_WEIGHTS = True
_C.MODEL.DECODER.QKV_BIAS = True
_C.MODEL.DECODER.NORM = 'LayerNorm'
_C.MODEL.DECODER.NORM_EPS = 1e-5
_C.MODEL.DECODER.MAX_POSITION_EMBEDDINGS = 4096
_C.MODEL.DECODER.EMBEDDDINGS_DROP_RATE = 0.1
_C.MODEL.DECODER.ATTENTION_DROPOUT_RATE = 0.1
_C.MODEL.DECODER.HIDDEN_DROPOUT_RATE = 0.1

# BottleNeck parameters
_C.MODEL.ADD_SEP_TOKEN = False

_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.HIDDEN_SIZE = 32

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
# Training by epochs
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20

# Training by iters
# _C.TRAIN.START_ITER = 0
# _C.TRAIN.ITERS = 300000
# _C.TRAIN.WARMUP_ITERS = 20000

_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = False
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# warmup_prefix used in CosineLRScheduler
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
# Gamma / Multi steps value, used in MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# Layer decay for fine-tuning
_C.TRAIN.LAYER_DECAY = 1.0

# Loss function
_C.TRAIN.LOSS_FUNCTION=['CE','Dice']
# Loss Weight
_C.TRAIN.LOSS_WEIGHT = None
# Auxiliary feature loss
_C.TRAIN.AUXILIARY_LOSS = False
# Auxiliary feature loss weight
_C.TRAIN.AUXILIARY_LOSS_WEIGHT = 0.1
# Loss mask type for training
_C.TRAIN.LOSS_MASK_TYPE = 'segment-predict'
# prompt number for loss mask
_C.TRAIN.PROMPT_NUMBER = 1
# Dice loss include background
_C.TRAIN.DICE_INCLUDE_BACKGROUND = False
# Dice loss use softmax
_C.TRAIN.DICE_SOFTMAX = True

# Skip iters to resume correctly when checkpoint is saved by iter
_C.TRAIN.SKIP_ITERS_TO_RESUME = 0

# # MoE
# _C.TRAIN.MOE = CN()
# # Only save model on master device
# _C.TRAIN.MOE.SAVE_MASTER = False

# TODO: Medical image augumentation settings


# TODO: Medical image test settings

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False
_C.TEST.SHUFFLE = False
_C.TEST.INCLUDE_BACKGROUND = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Attention mask and sequence length
_C.ATTENTION_MASK_TYPE = 'causal'
_C.SEQUENCE_LENGTH_PER_IMAGE = None
_C.SEQUENCE_LENGTH = None

_C.VAL_BY_ITER = False
_C.VAL_FREQ = 5000
_C.VAL_ITER = 200
_C.SAVE_BY_ITER = False
_C.SAVE_FREQ_BY_ITER = 5000

_C.OVERIDE_LR_SCHEDULER = False

# Enable Pytorch automatic mixed precision (amp).
_C.AMP_ENABLE = True
# [Deprecated] Mixed precision opt level of apex, if O0, no apex amp is used ('O0', 'O1', 'O2')
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
# for acceleration
_C.FUSED_WINDOW_PROCESS = False
_C.FUSED_LAYERNORM = False

# Tnfer test only, overwritten by command line argument
_C.INFER_TEST_MODE = False 

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    print(args.cfgs)
    for cfg in args.cfgs:
        _update_config_from_file(config, cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('data_path'):
        config.DATA.DATA_PATH = args.data_path
    if _check_args('zip'):
        config.DATA.ZIP_MODE = True
    if _check_args('cache_mode'):
        config.DATA.CACHE_MODE = args.cache_mode
    if _check_args('pretrained'):
        config.MODEL.PRETRAINED = args.pretrained
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args('amp_opt_level'):
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")
        if args.amp_opt_level == 'O0':
            config.AMP_ENABLE = False
    if _check_args('disable_amp'):
        config.AMP_ENABLE = False
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('tag'):
        config.TAG = args.tag
    if _check_args('eval'):
        config.EVAL_MODE = True
    if _check_args('throughput'):
        config.THROUGHPUT_MODE = True
    if _check_args('infer_test'):
        config.INFER_TEST_MODE = True

    # for acceleration
    if _check_args('fused_window_process'):
        config.FUSED_WINDOW_PROCESS = True
    if _check_args('fused_layernorm'):
        config.FUSED_LAYERNORM = True
    ## Overwrite optimizer if not None, currently we use it for [fused_adam, fused_lamb]
    if _check_args('optim'):
        config.TRAIN.OPTIMIZER.NAME = args.optim
        
    config.SEQUENCE_LENGTH_PER_IMAGE = (config.DATA.IMG_SIZE // config.MODEL.ENCODER.PATCH_SIZE) ** 2
    if 'fuse' not in config.ATTENTION_MASK_TYPE:
        config.SEQUENCE_LENGTH = 2 * config.DATA.PAIRS_NUM_PER_SEQUENCE * config.SEQUENCE_LENGTH_PER_IMAGE
    else:
        assert 'fuse' in config.TRAIN.LOSS_MASK_TYPE, "fuse attention mask should be used with fuse loss mask"
        config.SEQUENCE_LENGTH = config.DATA.PAIRS_NUM_PER_SEQUENCE * config.SEQUENCE_LENGTH_PER_IMAGE
    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config