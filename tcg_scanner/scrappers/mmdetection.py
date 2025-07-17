import mmcv
import mmengine
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
# Choose to use a config and initialize the detector
config_file = r'C:\Users\daforbes\Desktop\projects\tcg_scanner\dataset\mask_point\my_config.py'
# Setup a checkpoint file to load
checkpoint_file = r'C:\Users\daforbes\Desktop\projects\tcg_scanner\dataset\mask_point\best_coco_segm_mAP_epoch_13.pth'

# register all modules in mmdet into the registries
register_all_modules()

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

# Use the detector to do inference
image = mmcv.imread(r'C:\Users\daforbes\Desktop\projects\tcg_scanner\raw\downloaded_cards\xy-trainer-kit-latios\en_US-TK8A-014-machoke.jpg',channel_order='rgb')
result = inference_detector(model, image)
print(result.pred_instances)

from mmdet.registry import VISUALIZERS
# init visualizer(run the block only once in jupyter notebook)
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# the dataset_meta is loaded from the checkpoint and
# then pass to the model in init_detector
visualizer.dataset_meta = model.dataset_meta

# show the results
visualizer.add_datasample(
    'result',
    image,
    data_sample=result,
    draw_gt = None,
    draw_pred=True,       # Draw predictions           # Display the image directly
    wait_time=0,
)
visualizer.show()