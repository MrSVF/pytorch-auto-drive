import torchvision
import os
from PIL import Image

from ..transforms import functional as F, ToTensor


# Base class for lane image folder datasets (usually for visualizations)
class ImageFolderLaneBase(torchvision.datasets.VisionDataset):
    def __init__(self, root=None, root_output=None, transforms=None, keypoint_process_fn=None):
        super().__init__(root, transforms, None, None)
        self.output_dir = root_output
        os.makedirs(self.output_dir, exist_ok=True)
        self.filenames = []
        self.images = []
        self.keypoints = None
        self.gt_keypoints = None
        self.masks = None
        self.keypoint_process_fn = keypoint_process_fn

    def __getitem__(self, index):
        """Return transformed image / original image / save filename / labels (if exist)

        Args:
            index (int): Frame index

        Returns:
            array, array, dictionary: transformed image, original image, dict with keypoints
        """
        img = Image.open(self.images[index]).convert('RGB')
        if img.size==(2560, 1440):
            img = img.crop((560, 220, 2000, 1000)) # Crop frame to ensure detection
        elif img.size==(1920, 1080):
            img = img.crop((240, 150, 1680, 930)) # Crop frame to ensure detection

        filename = os.path.join(self.output_dir, self.filenames[index])
        original_img = F.to_tensor(img).clone()
        mask = None
        if self.masks is not None:
            w, h = F._get_image_size(img)
            mask = ToTensor.label_to_tensor(
                F.resize(Image.open(self.masks[index]), size=[h, w], interpolation=Image.NEAREST)
            )

        # Transforms
        if self.transforms is not None:
            img = self.transforms(img)

        # Process potential target
        keypoint = None
        gt_keypoint = None
        if self.keypoints is not None:
            keypoint = self.keypoint_process_fn(self.keypoints[index])
        if self.gt_keypoints is not None:
            gt_keypoint = self.keypoint_process_fn(self.gt_keypoints[index])

        return img, original_img, {
            'filename': filename,
            'keypoint': keypoint,
            'gt_keypoint': gt_keypoint,
            'mask': mask
        }

    def make_sub_dirs(self):
        # Make sub dirs
        for f in self.filenames:
            dir_name = os.path.join(self.output_dir, f[:f.rfind('/')])
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

    def __len__(self):
        return len(self.images)
