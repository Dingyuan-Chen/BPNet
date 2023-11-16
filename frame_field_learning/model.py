import torch
from torchvision.models.segmentation._utils import _SimpleSegmentationModel
from frame_field_learning import tta_utils

import torch.nn.functional as F

def get_out_channels(module):
    if hasattr(module, "out_channels"):
        return module.out_channels
    children = list(module.children())
    i = 1
    out_channels = None
    while out_channels is None and i <= len(children):
        last_child = children[-i]
        out_channels = get_out_channels(last_child)
        i += 1
    # If we get out of the loop but out_channels is None, then the prev child of the parent module will be checked, etc.
    return out_channels

def sampling_buffer(mask, N):
    """
    Follows 3.1. Point Selection for Inference and Training
    In Train:, `The sampling strategy selects N points on a feature map to train on.`
    In Inference, `then selects the N most uncertain points`
    Args:
        mask(Tensor): [B, C, H, W]
        N(int): `During training we sample as many points as there are on a stride 16 feature map of the input`
        k(int): Over generation multiplier
        beta(float): ratio of importance points
        training(bool): flag
    Return:
        selected_point(Tensor) : flattened indexing points [B, num_points, 2]
    """
    device = mask.device
    B, H, W = mask.shape
    H_step, W_step = 1 / H, 1 / W

    N = min(H * W, N)
    _, idx = mask.view(B, -1).topk(N, dim=1)

    points = torch.zeros(B, N, 2, dtype=torch.float, device=device)
    points[:, :, 0] = W_step / 2.0 + (idx % W).to(torch.float) * W_step
    points[:, :, 1] = H_step / 2.0 + (idx // W).to(torch.float) * H_step
    return idx, points

def point_sample(input, point_coords, **kwargs):
    """
    From Detectron2, point_features.py#19
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output

class FrameFieldModel(torch.nn.Module):
    def __init__(self, config: dict, backbone, train_transform=None, eval_transform=None):
        """

        :param config:
        :param backbone: A _SimpleSegmentationModel network, its output features will be used to compute seg and framefield.
        :param train_transform: transform applied to the inputs when self.training is True
        :param eval_transform: transform applied to the inputs when self.training is False
        """
        super(FrameFieldModel, self).__init__()
        assert config["compute_seg"] or config["compute_crossfield"], \
            "Model has to compute at least one of those:\n" \
            "\t- segmentation\n" \
            "\t- cross-field"
        assert isinstance(backbone, _SimpleSegmentationModel), \
            "backbone should be an instance of _SimpleSegmentationModel"
        self.config = config
        self.backbone = backbone
        self.train_transform = train_transform
        self.eval_transform = eval_transform

        backbone_out_features = get_out_channels(self.backbone)

        # --- Add other modules if activated in config:
        seg_channels = 0
        if self.config["compute_seg"]:
            seg_channels = self.config["seg_params"]["compute_vertex"]\
                           + self.config["seg_params"]["compute_edge"]\
                           + self.config["seg_params"]["compute_interior"]
            self.seg_module = torch.nn.Sequential(
                torch.nn.Conv2d(backbone_out_features, backbone_out_features, 3, padding=1),
                torch.nn.BatchNorm2d(backbone_out_features),
                torch.nn.ELU(),
                torch.nn.Conv2d(backbone_out_features, seg_channels, 1),
                torch.nn.Sigmoid(),)

        if self.config["compute_crossfield"]:
            crossfield_channels = 4
            self.crossfield_module = torch.nn.Sequential(
                torch.nn.Conv2d(backbone_out_features + seg_channels, backbone_out_features, 3, padding=1),
                torch.nn.BatchNorm2d(backbone_out_features),
                torch.nn.ELU(),
                torch.nn.Conv2d(backbone_out_features, crossfield_channels, 1),
                torch.nn.Tanh(),
            )

    def inference(self, xb):
        image = xb["image"]
        outputs = {}

        # --- Extract features for every pixel of the image with a U-Net --- #
        backbone_features = self.backbone(image)["out"]

        m = torch.nn.AdaptiveAvgPool2d((image.size(-2), image.size(-1)))

        if self.config["compute_seg"]:
            # --- Output a segmentation of the image --- #
            seg = self.seg_module(backbone_features)

            seg_to_cat = seg.clone().detach()
            outputs["seg"] = seg

            backbone_features = torch.cat([backbone_features, seg_to_cat], dim=1)  # Add seg to image features


        if self.config["compute_crossfield"]:
            # --- Output a cross-field of the image --- #
            crossfield = 2 * self.crossfield_module(backbone_features)  # Outputs c_0, c_2 values in [-2, 2]
            outputs["crossfield"] = crossfield

        return outputs

    # @profile
    def forward(self, xb, tta=False):
        # print("\n### --- PolyRefine.forward(xb) --- ####")
        if self.training:
            if self.train_transform is not None:
                xb = self.train_transform(xb)
        else:
            if self.eval_transform is not None:
                xb = self.eval_transform(xb)

        if not tta:
            final_outputs = self.inference(xb)
        else:
            final_outputs = tta_utils.tta_inference(self, xb, self.config["eval_params"]["seg_threshold"])

        return final_outputs, xb
