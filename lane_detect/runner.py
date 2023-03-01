import os
from pathlib import Path
import json
import cv2
import shutil

from .run_lane_img_dir import run as run_lane


ROOT = Path("adasys/ml_models/lane_detect")
weights_path = Path("weights")


class Base:

    def __init__(self):
        pass

    def read_frames_paths(self):
        self.frames_paths = list(filter(lambda x: x.endswith("png") or x.endswith("jpg"), os.listdir(self.path)))

    @staticmethod
    def read_frame(frame_path: str):
        img = cv2.imread(frame_path)
        return img

    @staticmethod
    def find_frame(filename, path) -> int:
        path = os.path.join(path, 'results/lane_detect/result.json')
        with open(path, "r") as predictions_f:
            predictions_list = json.load(predictions_f)
        for elem in predictions_list:
            for frame in elem:
                if frame == filename:
                    return elem[frame]
        return False

    def run(self, src_dir, model_key, version):
        camera = "FORWARD"
        result_dir_path = os.path.join(src_dir, camera, "results", model_key)
        self.path = Path(src_dir) / camera
        run_lane(
            weights_path=weights_path / "vgg16_scnn_tusimple_20210224.pt",  # model.pt path(s)
            config_path=ROOT / "pytorch_auto_drive/configs/lane_detection/scnn/vgg16_tusimple.py",
            path_to_folder_with_frames=self.path,  # file/dir/URL/glob, 0 for webcam
            result_path=Path(result_dir_path),  # save results to project/name
            name=f"result_model",  # save results to project/name
        )
        # for report
        report = []
        meta_path = list(filter(lambda x: x.endswith("json"), os.listdir(self.path)))[0]
        meta_path = os.path.join(self.path, meta_path)
        with open(meta_path, "r") as meta_f:
            meta_data = json.load(meta_f)
        self.result_dir_path = result_dir_path
        os.makedirs(result_dir_path, exist_ok=True)

        self.read_frames_paths()
        for frame_path in self.frames_paths:  # type: ignore
            prediction = self.find_frame(filename=frame_path, path=self.path)
            time_stamp = meta_data['data'][frame_path]
            report.append(
                {
                    "file": frame_path,
                    "prediction": prediction,
                    "time": time_stamp,
                    "model": model_key,
                    "model_version": version,
                    "camera": camera
                }
            )
        result_file_path = os.path.join(result_dir_path, f"version_{version}.json")
        model_result_path = os.path.join(self.path, 'results/lane_detect')
        shutil.rmtree(model_result_path)
        os.makedirs(result_dir_path, exist_ok=True)
        with open(result_file_path, "w") as f:
            json.dump(report, f)
        return report
