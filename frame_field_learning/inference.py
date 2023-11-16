import sys

from tqdm import tqdm
import scipy

import numpy as np
import torch

from . import local_utils
from . import polygonize

from lydorn_utils import image_utils
from lydorn_utils import print_utils
from lydorn_utils import python_utils

import segmentation_refinement as refine
import cv2

from torch_lydorn.torch.utils.complex import complex_mul


def network_inference(config, model, batch):
    batch = local_utils.batch_to_cuda(batch)
    pred, batch = model(batch, tta=config["eval_params"]["test_time_augmentation"])
    return pred, batch

def compute_field(crossfield_angle):
    """
    :param crossfield_angle:
    :return: N, W, H = 4, 224, 224
    """

    z = torch.cat([torch.cos(crossfield_angle),
                   torch.sin(crossfield_angle)], dim=0)
    z_90deg = torch.cat((- z[1:2, ...], z[0:1, ...]), dim=0)
    gt_field = torch.cat((z, z_90deg), dim=0)

    return gt_field

def uv_to_c0c2(uv: torch.Tensor) -> torch.Tensor:
    u, v = torch.chunk(uv, 2, dim=0)
    u_squared = complex_mul(u, u, complex_dim=0)
    v_squared = complex_mul(v, v, complex_dim=0)
    c0 = complex_mul(u_squared, v_squared, complex_dim=0)
    c2 = - (u_squared + v_squared)
    c0c2 = torch.cat([c0, c2], dim=0)
    return c0c2

def inference(config, model, tile_data, compute_polygonization=False, pool=None):
    if config["eval_params"]["patch_size"] is not None:
        # Cut image into patches for inference
        inference_with_patching(config, model, tile_data)
        single_sample = True
    else:
        # Feed images as-is to the model
        inference_no_patching(config, model, tile_data)
        single_sample = False

    # Polygonize:
    if compute_polygonization:
        pool = None if single_sample else pool  # A single big image is being processed

        model_dir = './models'
        refiner = refine.Refiner(device='cuda:0', model_folder=model_dir)

        # interior
        tile_data["seg"][:, 0, ...][tile_data["seg"][:, 0, ...] < tile_data["seg"][:, 1, ...]] = 0

        for idx, (name, seg) in enumerate(zip(tile_data['image_filepath'], tile_data["seg"])):
            image = cv2.imread(name)
            mask = 255. * (seg[0] > 0.5).cpu().numpy()
            output = refiner.refine(image, mask, fast=True, L=768)

            output = output / 255.
            tile_data["seg"][idx, 0, ...] = torch.tensor(output).to(tile_data["seg"].device)

        crossfield = tile_data["crossfield"] if "crossfield" in tile_data else None

        polygons_batch, probs_batch = polygonize.polygonize(config["polygonize_params"], tile_data["seg"], crossfield_batch=crossfield,
                                         pool=pool)
        tile_data["polygons"] = polygons_batch
        tile_data["polygon_probs"] = probs_batch

    return tile_data


def inference_no_patching(config, model, tile_data):
    with torch.no_grad():
        batch = {
            "image": tile_data["image"],
            "image_mean": tile_data["image_mean"],
            "image_std": tile_data["image_std"]
        }

        try:
            pred, batch = network_inference(config, model, batch)
        except RuntimeError as e:
            print_utils.print_error("ERROR: " + str(e))
            if 1 < config["optim_params"]["eval_batch_size"]:
                print_utils.print_info("INFO: Try lowering the effective batch_size (which is {} currently). "
                                       "Note that in eval mode, the effective bath_size is equal to double the batch_size "
                                       "because gradients do not need to "
                                       "be computed so double the memory is available. "
                                       "You can override the effective batch_size with the --eval_batch_size command-line argument."
                                       .format(config["optim_params"]["eval_batch_size"]))
            else:
                print_utils.print_info("INFO: The effective batch_size is 1 but the GPU still ran out of memory."
                                       "You can specify parameters to split the image into patches for inference:\n"
                                       "--eval_patch_size is the size of the patch and should be chosen as big as memory allows.\n"
                                       "--eval_patch_overlap (optional, default=200) adds overlaps between patches to avoid border artifacts."
                                       .format(config["optim_params"]["eval_batch_size"]))
            sys.exit()

        tile_data["seg"] = pred["seg"]
        if "crossfield" in pred:
            tile_data["crossfield"] = pred["crossfield"]

    return tile_data


def inference_with_patching(config, model, tile_data):
    assert len(tile_data["image"].shape) == 4 and tile_data["image"].shape[0] == 1, \
        f"When using inference with patching, tile_data should have a batch size of 1, " \
        f"with image's shape being (1, C, H, W), not {tile_data['image'].shape}"
    with torch.no_grad():
        # Init tile outputs (image is (N, C, H, W)):
        height = tile_data["image"].shape[2]
        width = tile_data["image"].shape[3]
        seg_channels = config["seg_params"]["compute_interior"] \
                       + config["seg_params"]["compute_edge"] \
                       + config["seg_params"]["compute_vertex"]
        if config["compute_seg"]:
            tile_data["seg"] = torch.zeros((1, seg_channels, height, width), device=config["device"])
        if config["compute_crossfield"]:
            tile_data["crossfield"] = torch.zeros((1, 4, height, width), device=config["device"])
        weight_map = torch.zeros((1, 1, height, width), device=config["device"])  # Count number of patches on top of each pixel

        # Split tile in patches:
        stride = config["eval_params"]["patch_size"] - config["eval_params"]["patch_overlap"]
        patch_boundingboxes = image_utils.compute_patch_boundingboxes((height, width),
                                                                      stride=stride,
                                                                      patch_res=config["eval_params"]["patch_size"])
        # Compute patch pixel weights to merge overlapping patches back together smoothly:
        patch_weights = np.ones((config["eval_params"]["patch_size"] + 2, config["eval_params"]["patch_size"] + 2),
                                dtype=np.float)
        patch_weights[0, :] = 0
        patch_weights[-1, :] = 0
        patch_weights[:, 0] = 0
        patch_weights[:, -1] = 0
        patch_weights = scipy.ndimage.distance_transform_edt(patch_weights)
        patch_weights = patch_weights[1:-1, 1:-1]
        patch_weights = torch.tensor(patch_weights, device=config["device"]).float()
        patch_weights = patch_weights[None, None, :, :]  # Adding batch and channels dims

        # Predict on each patch and save in outputs:
        for bbox in tqdm(patch_boundingboxes, desc="Running model on patches", leave=False):
            # Crop data
            batch = {
                "image": tile_data["image"][:, :, bbox[0]:bbox[2], bbox[1]:bbox[3]],
                "image_mean": tile_data["image_mean"],
                "image_std": tile_data["image_std"],
            }
            # Send batch to device
            try:
                pred, batch = network_inference(config, model, batch)
            except RuntimeError as e:
                print_utils.print_error("ERROR: " + str(e))
                print_utils.print_info("INFO: Reduce --eval_patch_size until the patch fits in memory.")
                raise e

            if config["compute_seg"]:
                tile_data["seg"][:, :, bbox[0]:bbox[2], bbox[1]:bbox[3]] += patch_weights * pred["seg"]
            if config["compute_crossfield"]:
                tile_data["crossfield"][:, :, bbox[0]:bbox[2], bbox[1]:bbox[3]] += patch_weights * pred["crossfield"]
            weight_map[:, :, bbox[0]:bbox[2], bbox[1]:bbox[3]] += patch_weights

        # Take care of overlapping parts
        if config["compute_seg"]:
            tile_data["seg"] /= weight_map
        if config["compute_crossfield"]:
            tile_data["crossfield"] /= weight_map

    return tile_data


def load_checkpoint(model, checkpoints_dirpath, device):
    """
    Loads best val checkpoint in checkpoints_dirpath
    """
    filepaths = python_utils.get_filepaths(checkpoints_dirpath, startswith_str="checkpoint.best_val.",
                                           endswith_str=".tar")
    if len(filepaths):
        filepaths = sorted(filepaths)
        filepath = filepaths[-1]  # Last best val checkpoint filepath in case there is more than one
        print_utils.print_info("Loading best val checkpoint: {}".format(filepath))
    else:
        # No best val checkpoint fount: find last checkpoint:
        filepaths = python_utils.get_filepaths(checkpoints_dirpath, endswith_str=".tar",
                                               startswith_str="checkpoint.")
        filepaths = sorted(filepaths)
        filepath = filepaths[-1]  # Last checkpoint
        print_utils.print_info("Loading last checkpoint: {}".format(filepath))

    device = torch.device(device)
    checkpoint = torch.load(filepath, map_location=device)  # map_location is used to load on current device

    model.load_state_dict(checkpoint['model_state_dict'])

    return model
