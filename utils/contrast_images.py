import os
import cv2

import argparse
import json
import numpy as np

import torch
from torch import device, load, no_grad
from torchvision.transforms.functional import crop

image = cv2.imread('/rebel/services/adasys/github/adasys/volumes/app_data/projects/1/processed/events/4/FORWARD2/2023_0323_070928_F_0000000199.jpg')

alpha = 1.5 # Contrast control (1.0-3.0)
beta = 0 # Brightness control (0-100)

adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

cv2.imshow('original', image)
cv2.imshow('adjusted', adjusted)
cv2.waitKey()


# class Base:
#     def __init__(self, path: str, path_out: str):
#         self._GPU: str = 'cuda:0'
#         self._CPU: str = 'cpu'

#         self.path = path
#         self.path_out = path_out

#         self.model = None
#         self.device = None

#         self.weights = None
#         # self.weights_path = 'src/models/lane/weights/weights.pt'
#         self.weights_path = 'src/lane_detector_service/trained_models/2022-12-8/efficientnet-v2-s-weights-100-epochs/15-22/80_15.pt'

#         self.frames_paths = None

#         self.report = {'results': {self.weights_path: []}}
#         # self.report = {}


#     def read_frames_paths(self):
#         self.frames_paths = os.listdir(self.path)

#     @staticmethod
#     def read_frame(frame_path: str):
#         return cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)


#     def run(self):
#         self.read_frames_paths()

#         if not os.path.exists(self.path_out):
#                 os.makedirs(self.path_out)

#         for frame_path in self.frames_paths:
#             frame = self.read_frame(frame_path=os.path.join(self.path, frame_path))
#             try:
#                 # frame = torch.tensor(np.expand_dims(np.transpose(frame, (2, 0, 1)), axis=0).astype(np.float32))
#                 # print('shape1:', (frame.shape))
#                 frame = frame[0:800, 960:1920]
#                 # print('shape2:', (frame.shape))
#                 # print('path_out:', os.path.join(self.path_out, frame_path))
#                 cv2.imwrite(os.path.join(self.path_out, frame_path), frame)
#             except:
#                 pass

#         return self

# '''
# Пример команды запуска:
# python /rebel/services/adasys/github/adasys/src/models/lane/crop_images.py --img_folder=/rebel/services/adasys/adasys_data/frames_retrieval_service/frames/line/20200904_175240 --res_folder=/rebel/services/adasys/adasys_data/frames_retrieval_service/frames/line/20200904_175240_cr
# '''

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='PytorchAutoDrive lane directory vis', conflict_handler='resolve')
#     parser.add_argument('--img_folder', type=str, help='Path to images folder', required=True)
#     parser.add_argument('--res_folder', type=str, help='Path to results folder', required=True)

#     args = parser.parse_args()

#     path_to_folder_with_frames = args.img_folder
#     path_to_folder_with_results = args.res_folder

#     runner = Base(path=path_to_folder_with_frames, path_out=path_to_folder_with_results).run()
