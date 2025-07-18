data_root = r"/home/user/Sync/pkmn_tracker/tcg_scanner/dataset"

# Dataset type
dataset_type = 'CocoDataset'

# Use a stronger base model with RefineMask (better for detailed masks)
_base_ = [
    r'/home/user/Sync/mmdetection/configs/mask_rcnn/mask-rcnn_r101_fpn_2x_coco.py'
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco/segm_mAP',  # 👈 save the best model based on segm score
        rule='greater'         # higher = better (default, but explicit)
    )
)

# Classes for detection
classes = ('pokemon card', 'psa label')

# Update model to increase mask quality
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=2),
        mask_head=dict(
            num_classes=2,
            loss_mask=dict(
                type='CrossEntropyLoss',
                use_mask=True,
                loss_weight=1.0
            )
        )
    )
)

# Training data
train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + r'/images/train/train_conf/new.json',
        data_prefix=dict(img=data_root + r'/images/train'),
        metainfo=dict(classes=classes)
    )
)

# Validation data
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + r'/images/val/val_conf/new.json',
        data_prefix=dict(img=data_root + r'/images/val'),
        metainfo=dict(classes=classes)
    )
)

# Training config
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=70, val_interval=1)

# Optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
)

# Evaluation
val_cfg = dict(type='ValLoop')
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + r'/images/val/val_conf/new.json',
    metric=['bbox', 'segm']
)

# Output directory
work_dir = "/home/user/Sync/workdir/"


# Enable automatic mixed precision (optional for faster training)
fp16 = dict(loss_scale='dynamic')
