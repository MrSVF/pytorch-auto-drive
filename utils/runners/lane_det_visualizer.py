import os
import torch
import cv2
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from abc import abstractmethod
if torch.__version__ >= '1.6.0':
    from torch.cuda.amp import autocast
else:
    from ..torch_amp_dummy import autocast

from torchvision.transforms.functional import crop
from .base import BaseVisualizer, BaseVideoVisualizer, get_collate_fn
from ..datasets import DATASETS
from ..transforms import TRANSFORMS
from ..lane_det_utils import lane_as_segmentation_inference
from ..vis_utils import lane_detection_visualize_batched, save_images

from utils.models.lane_detection.dashlane_detect import DashLaneDet


def lane_label_process_fn(label):
    # The CULane format
    # input: label txt file path or content as list
    if isinstance(label, str):
        with open(label, 'r') as f:
            label = f.readlines()
    target = []
    for line in label:
        temp = [float(x) for x in line.strip().split(' ')]
        target.append(np.array(temp).reshape(-1, 2))

    return target


class LaneDetVisualizer(BaseVisualizer):
    dataset_statistics = ['keypoint_color']
    color_pool = [[0, 0, 0],
                  [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                  [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                  [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                  [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                  [0, 0, 0]]

    @torch.no_grad()
    def lane_inference(self, images, original_size=None):
        cps = None  # Bézier control points
        if original_size is None:
            original_size = self._cfg['original_size']
        with autocast(self._cfg['mixed_precision']):
            if self._cfg['seg']:  # Seg methods
                keypoints = lane_as_segmentation_inference(self.model, images,
                                                           [self._cfg['input_size'], original_size],
                                                           self._cfg['gap'],
                                                           self._cfg['ppl'],
                                                           self._cfg['thresh'],
                                                           self._cfg['dataset_name'],
                                                           self._cfg['max_lane'])
            else:
                return_cps = self._cfg['style'] == 'bezier'
                res = self.model.inference(images,
                                           [self._cfg['input_size'], original_size],
                                           self._cfg['gap'],
                                           self._cfg['ppl'],
                                           self._cfg['dataset_name'],
                                           self._cfg['max_lane'],
                                           return_cps=return_cps)
                if return_cps:
                    cps, keypoints = res
                else:
                    keypoints = res

        return cps, [[np.array(lane) for lane in image] for image in keypoints]

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_loader(self, *args, **kwargs):
        pass


class LaneDetDir(LaneDetVisualizer):
    dataset_tensor_statistics = ['colors']

    def __init__(self, cfg):
        super().__init__(cfg)
        if self._cfg['save_path'] is not None:
            os.makedirs(self._cfg['save_path'], exist_ok=True)
        if self._cfg['use_color_pool']:
            self._cfg['colors'] = torch.tensor(self.color_pool,
                                               dtype=self._cfg['colors'].dtype,
                                               device=self._cfg['colors'].device)

    def get_loader(self, cfg):
        if 'vis_dataset' in cfg.keys():
            dataset_cfg = cfg['vis_dataset']
        else:
            dataset_cfg = dict(
                name='ImageFolderLaneDataset',
                root_image=self._cfg['image_path'],
                root_keypoint=self._cfg['keypoint_path'],
                root_gt_keypoint=self._cfg['gt_keypoint_path'],
                root_mask=self._cfg['mask_path'],
                root_output=self._cfg['save_path'],
                image_suffix=self._cfg['image_suffix'],
                keypoint_suffix=self._cfg['keypoint_suffix'],
                gt_keypoint_suffix=self._cfg['gt_keypoint_suffix'],
                mask_suffix=self._cfg['mask_suffix']
            )
        dataset = DATASETS.from_dict(dataset_cfg,
                                     transforms=TRANSFORMS.from_dict(cfg['test_augmentation']),
                                     keypoint_process_fn=lane_label_process_fn)
        collate_fn = get_collate_fn('dict_collate_fn')  # Use dicts for customized target
        # print('dataset:', len(dataset.images))
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=self._cfg['batch_size'],
                                                 collate_fn=collate_fn,
                                                 num_workers=self._cfg['workers'],
                                                 shuffle=False)

        return dataloader, cfg['dataset']['name']

    def run(self):
        dashline_detector = DashLaneDet()
        res_json =[]
        for imgs, original_imgs, targets in tqdm(self.dataloader):
            filenames = [i['filename'] for i in targets]
            res_filename = os.path.join(os.path.dirname(filenames[0]), 'result.json')
            keypoints = [i['keypoint'] for i in targets]
            gt_keypoints = [i['gt_keypoint'] for i in targets]
            masks = [i['mask'] for i in targets]
            cps = None
            if keypoints.count(None) == len(keypoints):
                keypoints = None
            if gt_keypoints.count(None) == len(gt_keypoints):
                gt_keypoints = None
            if masks.count(None) == len(masks):
                masks = None
            else:
                masks = torch.stack(masks)
            if self._cfg['pred']:  # Inference keypoints
                imgs_to_cv2 = original_imgs.numpy().transpose(0, 2, 3, 1)
                print('imgs_to_cv2:', imgs_to_cv2.sum())
                if masks is not None:
                    masks = masks.to(self.device)
                imgs = imgs.to(self.device)
                original_imgs = original_imgs.to(self.device)
                cps, keypoints = self.lane_inference(imgs, original_imgs.shape[2:])
                np_kps = np.array(keypoints)
                ind_red_lines = []
                fnames = []
                # koefs = []
                solidlines = []
                for i, fnname in zip(range(original_imgs.shape[0]), filenames):
                    np_kpt = np_kps[i]
                    cross = False
                    for j in range(len(np_kpt)):
                        np_kp = np_kpt[j]
                        np_kp_clear = np_kp[np_kp.min(axis=1)>=0,:]
                        x = np_kp_clear[:,0].reshape(-1, 1)
                        y = np_kp_clear[:,1].reshape(-1, 1)
                        reg = LinearRegression().fit(x, y)
                        k_koef = reg.coef_[0][0]
                        b_koef = reg.intercept_[0]
                        x1_red_line = 250
                        x2_red_line = original_imgs.shape[2:][1] - 100
                        y_red_line = original_imgs.shape[2:][0]
                        x_down_pos = (y_red_line - b_koef) / k_koef

                        if (x1_red_line < x_down_pos < x2_red_line) and np_kp_clear.shape[0] >= 3:
                            cross = True
                    if cross:
                        ind_red_lines.append(i)        
                        fnames.append(fnname)
                        # koefs.append((k_koef, b_koef))

                        isdashline = dashline_detector.detect_stoplineB(imgs_to_cv2[i], (k_koef, b_koef))
                        print('isdashline:', isdashline)
                        if not isdashline:
                            solidlines.append(fnname)
                    #     original_imgs[i] = original_imgs[i].clamp_(0.0, 1.0) * 255.0
                    #     original_imgs[i] = original_imgs[..., [2, 1, 0]][i].cpu().numpy().astype(np.uint8)        
                    #     cv2.line(original_imgs[i], (x1_red_line, y_red_line), (x2_red_line, y_red_line), color=(0, 0, 255), thickness=10)

            results = lane_detection_visualize_batched(original_imgs,
                                                       masks=masks,
                                                       keypoints=keypoints,
                                                       control_points=cps,
                                                       gt_keypoints=gt_keypoints,
                                                       mask_colors=self._cfg['colors'],
                                                       keypoint_color=self._cfg['keypoint_color'],
                                                       std=None, mean=None, style=self._cfg['style'],
                                                       compare_gt_metric=self._cfg['metric'],
                                                       idx_red_lines=ind_red_lines)
            save_images(results, filenames=filenames)
        
            res_json += [
                {file_name.split('/')[-1]: 1} if file_name in fnames \
                    else ({file_name.split('/')[-1]: 2} if file_name in solidlines else {file_name.split('/')[-1]: 0}) \
                        for file_name in filenames
                ]
        
        with open(res_filename, 'w') as f:
            json.dump(res_json, f)

class LaneDetVideo(BaseVideoVisualizer, LaneDetVisualizer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def run(self):
        # Must do inference
        for imgs, original_imgs in tqdm(self.dataloader):
            keypoints = None
            cps = None
            if self._cfg['pred']:
                imgs = imgs.to(self.device)
                original_imgs = original_imgs.to(self.device)
                cps, keypoints = self.lane_inference(imgs, original_imgs.shape[2:])
            results = lane_detection_visualize_batched(original_imgs,
                                                       masks=None,
                                                       keypoints=keypoints,
                                                       control_points=cps,
                                                       mask_colors=None,
                                                       keypoint_color=self._cfg['keypoint_color'],
                                                       std=None, mean=None, style=self._cfg['style'])
            results = results[..., [2, 1, 0]]
            for j in range(results.shape[0]):
                self.writer.write(results[j])


class LaneDetDataset(BaseVisualizer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def get_loader(self, cfg):
        pass

    def run(self):
        pass
