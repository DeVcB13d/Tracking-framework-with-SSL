import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import List, Sequence, Tuple, Union
from effdet import get_efficientdet_config, EfficientDet

class BarlowTwinsTransform:
    def __init__(
        self,
        train=True,
        input_height=224,
        gaussian_blur=True,
        jitter_strength=1.0,
        normalize=None,
    ):
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.jitter_strength = jitter_strength
        self.normalize = normalize
        self.train = train

        color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        color_transform = [
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.input_height)
            if kernel_size % 2 == 0:
                kernel_size += 1

            color_transform.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=kernel_size)], p=0.5
                )
            )

        self.color_transform = transforms.Compose(color_transform)

        if normalize is None:
            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose(
                [transforms.ToTensor(), normalize]
            )

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop((self.input_height, self.input_height)),
                transforms.RandomHorizontalFlip(p=0.5),
                self.color_transform,
                self.final_transform,
            ]
        )

        self.finetune_transform = None
        if self.train:
            self.finetune_transform = transforms.Compose(
                [
                    transforms.RandomCrop((32, 32), padding=4, padding_mode="reflect"),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.finetune_transform = transforms.Compose(
                [
                    transforms.Resize((self.input_height, self.input_height)),
                    transforms.ToTensor(),
                ]
            )

    def __call__(self, sample):
        return (
            self.transform(sample),
            self.transform(sample),
            self.finetune_transform(sample),
        )


class BarlowTwinsLoss(nn.Module):
    def __init__(self, batch_size, lambda_coeff=5e-3, z_dim=128):
        super().__init__()

        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=8192):
        super().__init__()

        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x):
        return self.projection_head(x)


class ReprNet(nn.Module):
    def __init__(self, config):
        super(ReprNet, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(config.num_levels):
            self.conv.append(
                nn.AvgPool2d(2 ** (config.max_level - i - 1)),
            )

    def forward(self, x: List[torch.Tensor]):
        outputs = []
        for level, x_l in enumerate(x):
            outputs.append(self.conv[level](x_l))
        return sum(outputs)

# creating an efficientdet model to be used in barlowtwins by removing final box and class layers
def create_model(num_classes=1, image_size=512, architecture="tf_efficientnetv2_l"):
    config = get_efficientdet_config(architecture)
    config.update({"num_classes": num_classes})
    config.update({"image_size": (image_size, image_size)})
    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = nn.Identity()
    net.box_net = nn.Identity()
    return net

