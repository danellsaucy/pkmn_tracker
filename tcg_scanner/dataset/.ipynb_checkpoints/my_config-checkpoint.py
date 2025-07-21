data_root = r"/home/user/Sync/pkmn_tracker/tcg_scanner/dataset"

# Dataset type
dataset_type = 'CocoDataset'

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,                          # Check every epoch
        save_best='coco/segm_mAP',           # Keep only best based on segm
        rule='greater',                      # Higher is better
        max_keep_ckpts=1,                    # ✅ Only keep 1 checkpoint
        save_last=False,                     # Don't keep last epoch checkpoint
        greater_keys=['coco/segm_mAP']       # (Optional) Ensure it ranks by segm
    )
)

# Use a stronger base model with RefineMask (better for detailed masks)
_base_ = [
    r'/home/user/Sync/mmdetection/configs/mask2former/mask2former_r101_8xb2-lsj-50e_coco-panoptic.py'
]

# Classes for detection
classes = ('pokemon card', 'psa label')

model = dict(
    type='Mask2Former',
    data_preprocessor=dict(
        pad_seg=False,                # ✅ disable semantic segmentation padding
        batch_augments=None           # ✅ remove augments that rely on sem seg
    )
)

train_dataloader = dict(
    _delete_=True,  # <
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + r'/images/train/train_conf/new.json',
        data_prefix=dict(img=data_root + r'/images/train/'),
        metainfo=dict(classes=classes),
        pipeline=[
            dict(type='LoadImageFromFile'),

            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),

            # dict(
            #     type='Albu',
            #     # ✅ Don't set is_check_shapes here — it's not valid in MMDet config.
            #     transforms=[
            #         dict(type='RandomBrightnessContrast', p=0.5),
            #         dict(type='GaussianBlur', p=0.3),
            #         dict(type='RGBShift', r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
            #     ],
            #     bbox_params=dict(
            #         type='BboxParams',
            #         format='pascal_voc',
            #         label_fields=['labels']  # matches what `keymap` refers to
            #     ),
            #     keymap={
            #         'img': 'image',
            #         'gt_bboxes': 'bboxes',
            #         'gt_bboxes_labels': 'labels',  # ✅ This must match the field in bbox_params
            #         'gt_masks': 'masks'
            #     }
            # ),
            dict(type='PackDetInputs')
        ]
    )
)
# Validation data
val_dataloader = dict(
    _delete_=True,  # <
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + r'/images/val/val_conf/new.json',
        data_prefix=dict(img=data_root + r'/images/val/'),
        metainfo=dict(classes=classes),
        pipeline=[
            dict(type='LoadImageFromFile'),
            # dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='PackDetInputs')
        ]
    )
)

# Training config
train_cfg = dict(type='EpochBasedTrainLoop', _delete_=True, max_epochs=35, val_interval=1)

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05
    ),
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.0),
            'absolute_pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'query_embed': dict(decay_mult=0.0)
        }
    )
)

# Evaluation
val_cfg = dict(type='ValLoop')
val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=data_root + r'/images/val/val_conf/new.json',
    metric='segm'
)

test_dataloader = val_dataloader

# Output directory
work_dir = "/home/user/Sync/workdir/"


# Enable automatic mixed precision (optional for faster training)
fp16 = dict(loss_scale='dynamic')
