data_root = r"C:\Users\daforbes\Desktop\projects\tcg_scanner\dataset"

dataset_type = 'COCODataset'

_base_ = [r'C:\Users\daforbes\Desktop\projects\mmdetection\configs\faster_rcnn\faster-rcnn_r50_fpn_1x_coco.py']

classes = ('pokemon card', 'psa label')

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=len(classes))
    )
)

# update paths
train_dataloader = dict(
    dataset=dict(
        ann_file=data_root + r'\train_conf\dataset.json',
        data_prefix=dict(img=data_root + r'\images\train'),
        metainfo=dict(classes=('pokemon card', 'psa label'))
    )
)

val_dataloader = dict(
    dataset=dict(
        ann_file=data_root + r'\val_conf\dataset.json',
        data_prefix=dict(img=data_root + r'\images\val'),
        metainfo=dict(classes=('pokemon card', 'psa label'))
    )
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=1)

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
)

val_cfg = dict(type='ValLoop')

val_evaluator = dict(type='CocoMetric', ann_file=data_root + r'\val_conf\dataset.json')

# output dir
work_dir = r"C:\Users\daforbes\Desktop\projects\tcg_scanner\work_dir\\"
