import os
import pathlib
import warnings

import skimage.io
from multiprocess import Pool
from functools import partial

import numpy as np
from pycocotools.coco import COCO
import shapely.geometry

from tqdm import tqdm

import torch

from lydorn_utils import print_utils
from lydorn_utils import python_utils

from torch_lydorn.torch.utils.data import Dataset as LydornDataset, makedirs, files_exist, __repr__

from torch_lydorn.torchvision.datasets import utils

import cv2

class MappingChallenge(LydornDataset):
    def __init__(self, root, transform=None, pre_transform=None, fold="train", small=False, pool_size=1):
        assert fold in ["train", "val", "test_images"], "Input fold={} should be in [\"train\", \"val\", \"test_images\"]".format(fold)
        if fold == "test_images":
            print_utils.print_error("ERROR: fold {} not yet implemented!".format(fold))
            exit()
        self.root = root
        self.fold = fold
        makedirs(self.processed_dir)
        self.small = small
        if self.small:
            print_utils.print_info("INFO: Using small version of the Mapping challenge dataset.")
        self.pool_size = pool_size

        self.coco = None
        self.image_id_list = self.load_image_ids()
        self.stats_filepath = os.path.join(self.processed_dir, "stats.pt")
        self.stats = None
        if os.path.exists(self.stats_filepath):
            self.stats = torch.load(self.stats_filepath)
        self.processed_flag_filepath = os.path.join(self.processed_dir, "processed-flag-small" if self.small else "processed-flag")

        super(MappingChallenge, self).__init__(root, transform, pre_transform)

    def load_image_ids(self):
        image_id_list_filepath = os.path.join(self.processed_dir, "image_id_list-small.json" if self.small else "image_id_list.json")
        if os.path.exists(image_id_list_filepath):
            image_id_list = python_utils.load_json(image_id_list_filepath)
        else:
            coco = self.get_coco()
            image_id_list = coco.getImgIds(catIds=coco.getCatIds())

        # Save for later so that the whole coco object doesn't have to be instantiated when just reading processed samples with multiple workers:
        python_utils.save_json(image_id_list_filepath, image_id_list)

        return image_id_list

    def get_coco(self):
        if self.coco is None:
            annotation_filename = "annotation-small.json" if self.small else "annotation.json"
            annotations_filepath = os.path.join(self.raw_dir, self.fold, annotation_filename)
            self.coco = COCO(annotations_filepath)
        return self.coco

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed', self.fold) # processed

    @property
    def processed_file_names(self):
        l = []
        for image_id in self.image_id_list:
            l.append(os.path.join("data_{:012d}.pt".format(image_id)))
        return l

    def __len__(self):
        return len(self.image_id_list)

    def _download(self):
        pass

    def download(self):
        pass

    def _process(self):
        f = os.path.join(self.processed_dir, 'pre_transform.pt')
        if os.path.exists(f) and torch.load(f) != __repr__(self.pre_transform):
            warnings.warn(
                'The `pre_transform` argument differs from the one used in '
                'the pre-processed version of this dataset. If you really '
                'want to make use of another pre-processing technique, make '
                'sure to delete `{}` first.'.format(self.processed_dir))
        f = os.path.join(self.processed_dir, 'pre_filter.pt')
        if os.path.exists(f) and torch.load(f) != __repr__(self.pre_filter):
            warnings.warn(
                'The `pre_filter` argument differs from the one used in the '
                'pre-processed version of this dataset. If you really want to '
                'make use of another pre-fitering technique, make sure to '
                'delete `{}` first.'.format(self.processed_dir))

        if os.path.exists(self.processed_flag_filepath):
            return

        print('Processing...')

        makedirs(self.processed_dir)
        self.process()

        path = os.path.join(self.processed_dir, 'pre_transform.pt')
        torch.save(__repr__(self.pre_transform), path)
        path = os.path.join(self.processed_dir, 'pre_filter.pt')
        torch.save(__repr__(self.pre_filter), path)

        print('Done!')

    def process(self):
        images_relative_dirpath = os.path.join("raw", self.fold, "images")

        image_info_list = []
        coco = self.get_coco()
        for image_id in self.image_id_list:
            filename = coco.loadImgs(image_id)[0]["file_name"]

            # if filename in ["2.tif"]:
            #     continue
            annotation_ids = coco.getAnnIds(imgIds=image_id)
            annotation_list = coco.loadAnns(annotation_ids)
            image_info = {
                "image_id": image_id,
                "image_filepath": os.path.join(self.root, images_relative_dirpath, filename),
                "image_relative_filepath": os.path.join(images_relative_dirpath, filename),
                "annotation_list": annotation_list
            }
            image = skimage.io.imread(image_info["image_filepath"])
            image_info_list.append(image_info)

        sample_stats_list = []
        for im_info in tqdm(image_info_list):
            sample_stats_list.append(
                preprocess_one(im_info, self.pre_filter, self.pre_transform, self.processed_dir)
            )
        # Aggregate sample_stats_list
        image_s0_list, image_s1_list, image_s2_list, class_freq_list = zip(*sample_stats_list)
        image_s0_array = np.stack(image_s0_list, axis=0)
        image_s1_array = np.stack(image_s1_list, axis=0)
        image_s2_array = np.stack(image_s2_list, axis=0)
        class_freq_array = np.stack(class_freq_list, axis=0)

        image_s0_total = np.sum(image_s0_array, axis=0)
        image_s1_total = np.sum(image_s1_array, axis=0)
        image_s2_total = np.sum(image_s2_array, axis=0)

        image_mean = image_s1_total / image_s0_total
        image_std = np.sqrt(image_s2_total/image_s0_total - np.power(image_mean, 2))
        class_freq = np.sum(class_freq_array*image_s0_array[:, None], axis=0) / image_s0_total

        # Save aggregated stats
        self.stats = {
            "image_mean": image_mean,
            "image_std": image_std,
            "class_freq": class_freq,
        }
        torch.save(self.stats, self.stats_filepath)

        # Indicates that processing has been performed:
        pathlib.Path(self.processed_flag_filepath).touch()

    def get(self, idx):
        image_id = self.image_id_list[idx]
        data = torch.load(os.path.join(self.processed_dir, "data_{:012d}.pt".format(image_id)))

        data["image_mean"] = self.stats["image_mean"]
        data["image_std"] = self.stats["image_std"]
        data["class_freq"] = self.stats["class_freq"]
        return data

def random_crop(image, mask, new_h=650, new_w=650):
    h, w = image.shape[:2]

    y = np.random.randint(0, h - new_h)
    x = np.random.randint(0, w - new_w)

    image = image[y:y + new_h, x:x + new_w, :]
    mask = mask[y:y + new_h, x:x + new_w, :]

    return image, mask

def polygonFromMask(maskedArr):
    # adapted from https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    valid_poly = 0
    for contour in contours:
    # Valid polygons have >= 6 coordinates (3 points)
     if contour.size >= 6:
        segmentation.append(contour.astype(float).flatten().tolist())
        valid_poly += 1
    if valid_poly == 0:
     raise ValueError
    return segmentation

def preprocess_one(image_info, pre_filter, pre_transform, processed_dir):
    out_filepath = os.path.join(processed_dir, "data_{:012d}.pt".format(image_info["image_id"]))
    data = None
    if os.path.exists(out_filepath):
        # Load already-processed sample
        try:
            data = torch.load(out_filepath)
        except EOFError:
            pass

    if data is None:
        # Process sample:
        image = skimage.io.imread(image_info["image_filepath"])
        gt_polygons = []
        for annotation in image_info["annotation_list"]:
            flattened_segmentation_list = annotation["segmentation"]

            if type(flattened_segmentation_list) == dict:
                continue

            if len(flattened_segmentation_list) != 1:
                tmp_list = []
                for i in range(len(flattened_segmentation_list)):
                    tmp_list.extend(flattened_segmentation_list[i])
                flattened_segmentation_list = [tmp_list]
            if len(flattened_segmentation_list) != 1:
                print("WHAT!?!, len(flattened_segmentation_list = {}".format(len(flattened_segmentation_list)))
                print("To implement: if more than one segmentation in flattened_segmentation_list (MS COCO format), does it mean it is a MultiPolygon or a Polygon with holes?")
                raise NotImplementedError
            flattened_arrays = np.array(flattened_segmentation_list)
            coords = np.reshape(flattened_arrays, (-1, 2))
            polygon = shapely.geometry.Polygon(coords)

            # Filter out degenerate polygons (area is lower than 2.0)
            if 2.0 < polygon.area:
                gt_polygons.append(polygon)

        data = {
            "image": image,
            "gt_polygons": gt_polygons,
            "image_relative_filepath": image_info["image_relative_filepath"],
            "name": os.path.splitext(os.path.basename(image_info["image_relative_filepath"]))[0],
            "image_id": image_info["image_id"]
        }

        if pre_filter is not None and not pre_filter(data):
            return

        if pre_transform is not None:
            data = pre_transform(data)

        # masked_angles = data["gt_crossfield_angle"].astype(np.float) * data["gt_polygons_image"][:, :, 1].astype(np.float)
        # skimage.io.imsave("gt_crossfield_angle.png", data["gt_crossfield_angle"])
        # skimage.io.imsave("masked_angles.png", masked_angles)
        # exit()

        torch.save(data, out_filepath)

    # Compute stats for later aggregation for the whole dataset
    normed_image = data["image"] / 255
    image_s0 = data["image"].shape[0] * data["image"].shape[1]  # Number of pixels
    image_s1 = np.sum(normed_image, axis=(0, 1))  # Sum of pixel normalized values
    image_s2 = np.sum(np.power(normed_image, 2), axis=(0, 1))
    class_freq = np.mean(data["gt_polygons_image"], axis=(0, 1)) / 255

    return image_s0, image_s1, image_s2, class_freq