'''
Script to pretrain the model using barlow twins

'''
from .barlow import *
from train import PigsDatasetAdapter
from pathlib import Path
from functools import partial
from typing import Sequence, Tuple, Union
from pytorch_lightning import Callback, LightningModule, Trainer, LightningDataModule
from torchmetrics.functional import accuracy
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import torch
import os

def fn(warmup_steps, step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    else:
        return 1.0


def linear_warmup_decay(warmup_steps):
    return partial(fn, warmup_steps)


class BarlowTwins(LightningModule):
    def __init__(
        self,
        encoder,
        encoder_out_dim,
        get_repr,
        num_training_samples,
        batch_size,
        lambda_coeff=5e-3,
        z_dim=128,
        learning_rate=1e-4,
        warmup_epochs=10,
        max_epochs=200,
    ):
        super().__init__()

        self.encoder = encoder
        self.get_repr = get_repr
        self.projection_head = ProjectionHead(
            input_dim=encoder_out_dim, output_dim=z_dim
        )
        self.loss_fn = BarlowTwinsLoss(
            batch_size=batch_size, lambda_coeff=lambda_coeff, z_dim=z_dim
        )

        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.train_iters_per_epoch = num_training_samples // batch_size

    def forward(self, x):
        x = self.encoder(x)[0]
        x = self.get_repr(x)
        return x

    def shared_step(self, batch):
        (x1, x2, _), _ = batch

        x1 = self.encoder(x1)[0]
        x2 = self.encoder(x2)[0]

        x1 = self.get_repr(x1).squeeze()
        x2 = self.get_repr(x2).squeeze()

        z1 = self.projection_head(x1)
        z2 = self.projection_head(x2)

        return self.loss_fn(z1, z2)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]


def imagenet_normalization():
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [0.485, 0.456, 0.406]],
        std=[x / 255.0 for x in [0.229, 0.224, 0.225]],
    )
    return normalize


class PigsDatasetAdapter:
    def __init__(self, images_path):
        self.image_paths = [images_path / path for path in os.listdir(images_path)]

    def __len__(self):
        return len(self.image_paths)

    def get_image_by_idx(self, index):
        image = Image.open(self.image_paths[index])
        return image


class BarlowTwinsDataset(Dataset):
    def __init__(self, dataset_adaptor, transforms):
        self.ds = dataset_adaptor
        self.transforms = transforms

    def __getitem__(self, index):
        image = self.ds.get_image_by_idx(index)
        return *self.transforms(image), index

    def __len__(self):
        return len(self.ds)


class BarlowTwinsDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset_adaptor,
        validation_dataset_adaptor,
        train_transforms,
        valid_transforms,
        num_workers=4,
        batch_size=8,
    ):
        self.train_ds = train_dataset_adaptor
        self.valid_ds = validation_dataset_adaptor
        self.train_tfms = train_transforms
        self.valid_tfms = valid_transforms
        self.num_workers = num_workers
        self.batch_size = batch_size
        super().__init__()

    def train_dataset(self) -> BarlowTwinsDataset:
        return BarlowTwinsDataset(
            dataset_adaptor=self.train_ds, transforms=self.train_tfms
        )

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return train_loader

    def val_dataset(self) -> BarlowTwinsDataset:
        return BarlowTwinsDataset(
            dataset_adaptor=self.valid_ds, transforms=self.valid_tfms
        )

    def val_dataloader(self) -> DataLoader:
        valid_dataset = self.val_dataset()
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return valid_loader

    @staticmethod
    def collate_fn(batch):
        x1, x2, x3, image_ids = tuple(zip(*batch))
        x1, x2, x3 = (
            torch.stack(x1).float(),
            torch.stack(x2).float(),
            torch.stack(x3).float(),
        )

        return (x1, x2, x3), torch.tensor(image_ids)


class OnlineFineTuner(Callback):
    def __init__(
        self,
        encoder_output_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.optimizer: torch.optim.Optimizer

        self.encoder_output_dim = encoder_output_dim
        self.num_classes = num_classes

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # add linear_eval layer and optimizer
        pl_module.online_finetuner = nn.Linear(
            self.encoder_output_dim, self.num_classes
        ).to(pl_module.device)
        self.optimizer = torch.optim.Adam(
            pl_module.online_finetuner.parameters(), lr=1e-4
        )

    def extract_online_finetuning_view(
        self, batch: Sequence, device: Union[str, torch.device]
    ) -> Tuple[Tensor, Tensor]:
        (_, _, finetune_view), y = batch
        finetune_view = finetune_view.to(device)
        y = y.to(device)

        return finetune_view, y

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        x, y = self.extract_online_finetuning_view(batch, pl_module.device)

        with torch.no_grad():
            feats = pl_module(x)

        feats = feats.detach()
        preds = pl_module.online_finetuner(feats)
        loss = F.cross_entropy(preds, y)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        acc = accuracy(F.softmax(preds, dim=1), y)
        pl_module.log("online_train_acc", acc, on_step=True, on_epoch=False)
        pl_module.log("online_train_loss", loss, on_step=True, on_epoch=False)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        x, y = self.extract_online_finetuning_view(batch, pl_module.device)

        with torch.no_grad():
            feats = pl_module(x)

        feats = feats.detach()
        preds = pl_module.online_finetuner(feats)
        loss = F.cross_entropy(preds, y)

        acc = accuracy(F.softmax(preds, dim=1), y)
        pl_module.log(
            "online_val_acc", acc, on_step=False, on_epoch=True, sync_dist=True
        )
        pl_module.log(
            "online_val_loss", loss, on_step=False, on_epoch=True, sync_dist=True
        )


if __name__ == "__main__":
    batch_size = 6
    num_workers = 8
    max_epochs = 20
    z_dim = 4096

    barlow_path = Path(r"C:\Users\fayaz\NTNU\barlow\train_images")
    dataset_path = Path(r"C:\Users\fayaz\NTNU\Norsvin\Norsvin")

    train_images_path = dataset_path / "train/images"
    val_images_path = dataset_path / "val/images"

    pigs_train_ds = PigsDatasetAdapter(barlow_path)
    pigs_val_ds = PigsDatasetAdapter(val_images_path)

    architecture = "tf_efficientdet_d1"
    img_size = (640, 640)
    enc_out_dim = {"tf_efficientdet_d0": 64, "tf_efficientdet_d1": 88}

    train_transform = BarlowTwinsTransform(
        train=True,
        input_height=img_size[0],
        gaussian_blur=False,
        jitter_strength=0.5,
        normalize=imagenet_normalization(),
    )

    val_transform = BarlowTwinsTransform(
        train=False,
        input_height=img_size[0],
        gaussian_blur=False,
        jitter_strength=0.5,
        normalize=imagenet_normalization(),
    )

    dm = BarlowTwinsDataModule(
        train_dataset_adaptor=pigs_train_ds,
        validation_dataset_adaptor=pigs_val_ds,
        train_transforms=train_transform,
        valid_transforms=val_transform,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    encoder_out_dim = enc_out_dim[architecture]

    encoder = create_model(architecture=architecture, image_size=img_size[0])

    model = BarlowTwins(
        encoder=encoder,
        encoder_out_dim=encoder_out_dim,
        get_repr=ReprNet(config=get_efficientdet_config(architecture)),
        num_training_samples=len(pigs_train_ds),
        batch_size=batch_size,
        z_dim=z_dim,
    )

    out = model(torch.ones((1, 3, img_size[0], img_size[1])))

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=[0],
        accumulate_grad_batches=8,
    )

    trainer.fit(model, dm)

    torch.save(
        encoder.state_dict(), f"results/weights/barlow_pretrained_weights_backbone"
    )