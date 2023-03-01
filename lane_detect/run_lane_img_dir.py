import os
import sys
from pathlib import Path

import subprocess
from subprocess import PIPE

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # pytorch_auto_drive root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

'''
Примеры:
python pytorch_auto_drive/tools/vis/lane_img_dir.py --image-path=/rebel/services/adasys/adasys_data/frames_retrieval_service/frames/line/20200904_175240_cr --image-suffix=.png --save-path=/rebel/services/adasys/adasys_data/frames_retrieval_service/frames/line/vgg16_cul_tusi_cr_rg --config=pytorch_auto_drive/configs/lane_detection/scnn/vgg16_tusimple.py --checkpoint=pytorch_auto_drive/checkpoints/vgg16_scnn_tusimple_20210224.pt --workers=0 --pred
python adasys/ml_models/lane_detect/pytorch_auto_drive/tools/vis/lane_img_dir.py --image-path=/data/projects/2/processed/events/2/FORWARD --image-suffix=.png --save-path=/data/projects/2/processed/events/2/FORWARD/results/lane_detect --config=adasys/ml_models/lane_detect/pytorch_auto_drive/configs/lane_detection/scnn/vgg16_tusimple.py --checkpoint=adasys/ml_models/lane_detect/pytorch_auto_drive/checkpoints/vgg16_scnn_tusimple_20210224.pt --workers=0 --pred
'''

def run(
        weights_path=ROOT / "weights/vgg16_scnn_tusimple_20210224.pt",  # model.pt path(s)
        config_path=ROOT / "pytorch_auto_drive/configs/lane_detection/scnn/vgg16_tusimple.py",
        path_to_folder_with_frames=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        result_path=ROOT / "runs/detect",  # save results to project/name
        name="exp",  # save results to project/name
    ):

    picture_file_extension = ''
    list_extensions = ["png", "jpg"]

    for ext in list_extensions:
        if len(list(filter(lambda x: x.endswith(ext), os.listdir(path_to_folder_with_frames)))) > 0:
            picture_file_extension = ext
            break

    print(subprocess.run([
                        "python", 
                        "adasys/ml_models/lane_detect/pytorch_auto_drive/tools/vis/lane_img_dir.py",
                        "--image-path=" + str(path_to_folder_with_frames),
                        "--image-suffix=." + picture_file_extension,
                        "--save-path=" + str(result_path),
                        "--config=" + str(config_path), #adasys/ml_models/lane_detect/pytorch_auto_drive/configs/lane_detection/scnn/vgg16_tusimple.py",
                        "--checkpoint=" + str(weights_path), #adasys/ml_models/lane_detect/pytorch_auto_drive/checkpoints/vgg16_scnn_tusimple_20210224.pt",
                        "--workers=0",
                        "--pred"
                        ], check=True, stdout=PIPE).stdout)
