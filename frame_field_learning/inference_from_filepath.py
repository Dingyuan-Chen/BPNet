import os
import numpy as np
import skimage.io
import torch

from tqdm import tqdm

from . import data_transforms, save_utils
from .model import FrameFieldModel
from . import inference
from . import local_utils

from torch_lydorn import torchvision

from lydorn_utils import print_utils
from lydorn_utils import run_utils


def inference_from_filepath(config, in_filepaths, backbone, out_dirpath=None):
    # --- Online transform performed on the device (GPU):
    eval_online_cuda_transform = data_transforms.get_eval_online_cuda_transform(config)

    print("Loading model...")
    model = FrameFieldModel(config, backbone=backbone, eval_transform=eval_online_cuda_transform)
    model.to(config["device"])
    checkpoints_dirpath = run_utils.setup_run_subdir(config["eval_params"]["run_dirpath"], config["optim_params"]["checkpoints_dirname"])
    model = inference.load_checkpoint(model, checkpoints_dirpath, config["device"])
    model.eval()

    # Read image
    img_lst = os.listdir(in_filepaths[0])

    for _name in tqdm(img_lst):
        in_filepath = os.path.join(in_filepaths[0], _name)
        image = skimage.io.imread(in_filepath)

        if 3 < image.shape[2]:
            print_utils.print_info(f"Image {in_filepath} has more than 3 channels. Keeping the first 3 channels and discarding the rest...")
            image = image[:, :, :3]
        elif image.shape[2] < 3:
            print_utils.print_error(f"Image {in_filepath} has only {image.shape[2]} channels but the network expects 3 channels.")
            raise ValueError
        image_float = image / 255
        mean = np.mean(image_float.reshape(-1, image_float.shape[-1]), axis=0)
        std = np.std(image_float.reshape(-1, image_float.shape[-1]), axis=0)
        sample = {
            "image": torchvision.transforms.functional.to_tensor(image)[None, ...],
            "image_mean": torch.from_numpy(mean)[None, ...],
            "image_std": torch.from_numpy(std)[None, ...],
            "image_filepath": [in_filepath],
        }

        # pbar.set_postfix(status="Inference")
        tile_data = inference.inference(config, model, sample, compute_polygonization=True)

        tile_data = local_utils.batch_to_cpu(tile_data)

        # Remove batch dim:
        tile_data = local_utils.split_batch(tile_data)[0]

        # --- Saving outputs --- #

        # pbar.set_postfix(status="Saving output")

        # Figuring out_base_filepath out:
        if out_dirpath is None:
            out_dirpath = os.path.dirname(in_filepath)
        else:
            os.makedirs(out_dirpath, exist_ok=True)

        base_filename = os.path.splitext(os.path.basename(in_filepath))[0]
        out_base_filepath = (out_dirpath, base_filename)

        config["eval_params"]["save_individual_outputs"]["seg_mask"] = True
        config["eval_params"]["save_individual_outputs"]["seg"] = True
        config["eval_params"]["save_individual_outputs"]["poly_viz"] = True
        config["eval_params"]["save_individual_outputs"]["crossfield"] = False
        config["eval_params"]["save_individual_outputs"]["uv_angles"] = False
        config["eval_params"]["save_individual_outputs"]["poly_shapefile"] = False

        config["compute_seg"] = True
        if config["compute_seg"]:
            if config["eval_params"]["save_individual_outputs"]["seg_mask"]:
                seg_mask = 0.5 < tile_data["seg"][0]
                save_utils.save_seg_mask(seg_mask, out_base_filepath, "mask", tile_data["image_filepath"])

            if config["eval_params"]["save_individual_outputs"]["seg"]:
                save_utils.save_seg(tile_data["seg"], out_base_filepath, "seg", tile_data["image_filepath"])

            if config["eval_params"]["save_individual_outputs"]["seg_luxcarta"]:
                save_utils.save_seg_luxcarta_format(tile_data["seg"], out_base_filepath, "seg_luxcarta_format", tile_data["image_filepath"])

        if config["compute_crossfield"] and config["eval_params"]["save_individual_outputs"]["crossfield"]:
            save_utils.save_crossfield(tile_data["crossfield"], out_base_filepath, "crossfield")

        if config["eval_params"]["save_individual_outputs"]["uv_angles"]:
            save_utils.save_uv_angles(tile_data["crossfield"], out_base_filepath, "uv_angles", tile_data["image_filepath"])

        if "poly_viz" in config["eval_params"]["save_individual_outputs"] and \
                config["eval_params"]["save_individual_outputs"]["poly_viz"]:
            save_utils.save_poly_viz(tile_data["image"], tile_data["polygons"], tile_data["polygon_probs"], out_base_filepath, "poly_viz")
        if config["eval_params"]["save_individual_outputs"]["poly_shapefile"]:
            save_utils.save_shapefile(tile_data["polygons"], out_base_filepath, "poly_shapefile", tile_data["image_filepath"])