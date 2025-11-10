from torch import optim

from datasets.coco import CocoDetection
from transforms import presets
from optimizer import param_dict

# Commonly changed training configurations
num_epochs = 12   # train epochs
batch_size = 1    # total_batch_size = #GPU x batch_size
num_workers = 4   # workers for pytorch DataLoader
pin_memory = True # whether pin_memory for pytorch DataLoader
print_freq = 50   # frequency to print logs
starting_epoch = 0
max_norm = 0.1    # clip gradient norm
output_dir = None  # path to save checkpoints, default for None: checkpoints/{model_name}
find_unused_parameters = False  # useful for debugging distributed training

coco_path = "data/coco"  # /PATH/TO/YOUR/COCODIRs
train_transform = presets.detr  # see transforms/presets to choose a transform
train_dataset = CocoDetection(
    img_folder=coco_path+'train2017/',
    ann_file=coco_path+'annotations/instances_train2017.json',
    transforms=train_transform,
    train=True,
)
test_dataset = CocoDetection(
    img_folder=coco_path+'val2017/',
    ann_file=coco_path+'annotations/instances_val2017.json',
    transforms=None,  # the eval_transform is integrated in the model
)

# model config to train
model_path = "configs/position_detr/position_detr_resnet50.py"

learning_rate = 1e-4  # initial learning rate
optimizer = optim.AdamW(lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
lr_scheduler = optim.lr_scheduler.MultiStepLR(milestones=[10], gamma=0.1)

# This define parameter groups with different learning rate
param_dicts = param_dict.finetune_backbone_and_linear_projection(lr=learning_rate)
