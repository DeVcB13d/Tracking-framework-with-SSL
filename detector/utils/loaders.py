import torch
from torch.utils.data import DataLoader
from pathlib import Path
import PIL
import os


from torch.utils.data import Dataset

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import time
import numpy as np

from box_utils import *
from transforms import *

from pycocotools.coco import COCO


'''
Dataset adaptor for our norsvin dataset
'''
# To check if the image exists in the directory
# Creating a dataset adaptor
class PigsDatasetAdaptor:
    def __init__(self, images_dir_path, annotations_dataframe_path):
        self.images_dir_path = Path(images_dir_path)
        self.annotations_df_path =annotations_dataframe_path
        self.images = [image[:-5] for image in os.listdir(self.annotations_df_path)]
        # Loading the COCO objects
        self.annotations_dfs = self.load_coco_objs()
    
    def load_coco_objs(self) -> dict:
        # Load the coco annotations into a single dictionary
        # The index of the dictionary would be the image name
        print("Loading annotations into memory : ")
        tic = time.time()
        dfs = dict()
        self.annotations_df_list = os.listdir(self.annotations_df_path)
        for i in self.annotations_df_list:
            index_name = i[:-5]
            dfs[index_name] = COCO(f"{self.annotations_df_path}/{i}")
        print('Done (t={:0.2f}s)'.format(time.time()- tic))
        return dfs
        
    
    def __len__(self) -> int:
        return len(self.images)

    def get_image_and_labels_by_idx(self, index):
        '''
        This method would return
        - image : A Pil image
        - bboxes : a numpy array of shape [N,4] with ground truth values
        - class_labels : A numpy array of shape N having ground truth labels (pig here)
        - image_id : a unique identifier to identify the image
        
        '''
        img_name = self.images[index]
        # The coco object corresponding to the index
        coco_img_obj = self.annotations_dfs[img_name]
        # Getting the image IDs and loading them
        imgIds = coco_img_obj.getImgIds()
        imgs = coco_img_obj.loadImgs(imgIds)
        image = PIL.Image.open(f"{self.images_dir_path}/{imgs[0]['file_name']}")
        annIds = coco_img_obj.getAnnIds(imgIds=[imgs[0]['id']])
        anns = coco_img_obj.loadAnns(annIds)
        # Loading the bounding boxes
        pascal_bboxes = [get_pascal_bbox_list(i['bbox']) for i in anns]
        for i,box in enumerate(pascal_bboxes):
            if not check_bboxes(box):
                pascal_bboxes.pop(i)
        class_labels = np.ones(len(pascal_bboxes))
        return image, pascal_bboxes, class_labels, index
    
    def show_image(self, index):
        image, bboxes, class_labels, image_id = self.get_image_and_labels_by_idx(index)
        print(f"image_id: {image_id}")
        show_image(image, bboxes)
        print(class_labels)

'''
Creating the Dataset for training and testing
'''
class EfficientDetDataset(Dataset):
    def __init__(
        self, dataset_adaptor, transforms=get_valid_transforms()
    ):
        self.ds = dataset_adaptor
        self.transforms = transforms

    def __getitem__(self, index):
        (
            image,
            pascal_bboxes,
            class_labels,
            image_id,
        ) = self.ds.get_image_and_labels_by_idx(index)
        #show_image(image,pascal_bboxes)
        sample = {
            "image": np.array(image, dtype=np.float32),
            "bboxes": pascal_bboxes,
            "labels": class_labels,
        }
        #show_image(sample["image"])
        sample = self.transforms(**sample)
        sample["bboxes"] = np.array(sample["bboxes"])
        image = sample["image"]
        pascal_bboxes = sample["bboxes"]
        labels = sample["labels"]

        _, new_h, new_w = image.shape
        sample["bboxes"][:, [0, 1, 2, 3]] = sample["bboxes"][
            :, [1, 0, 3, 2]
        ]  # convert to yxyx

        target = {
            "bboxes": torch.as_tensor(sample["bboxes"], dtype=torch.float32),
            "labels": torch.as_tensor(labels),
            "image_id": torch.tensor([image_id]),
            "img_size": (new_h, new_w),
            "img_scale": torch.tensor([1.0]),
        }
        #exit()
        return image, target, image_id

    def __len__(self):
        return len(self.ds)

'''
Instead of creating seperate DataLoaders, using pytorch lightning we can create a data module 
This would be more helpful during predictions and training
'''
class EfficientDetDataModule(LightningDataModule):
    
    def __init__(self,
                img_size,
                train_dataset_adaptor,
                validation_dataset_adaptor,
                train_transforms=get_train_transforms,
                valid_transforms=get_valid_transforms,
                num_workers=4,
                batch_size=2):
        
        self.train_ds = train_dataset_adaptor
        self.valid_ds = validation_dataset_adaptor
        self.train_tfms = train_transforms(img_size)
        self.valid_tfms = valid_transforms(img_size)
        self.num_workers = num_workers
        self.batch_size = batch_size
        super().__init__()

    def train_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(
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
            collate_fn=self.collate_fn
        )

        return train_loader

    def val_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(
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
        images, targets, image_ids = tuple(zip(*batch))
        images = torch.stack(images)
        images = images.float()

        boxes = [target["bboxes"].float() for target in targets]
        labels = [target["labels"].float() for target in targets]
        img_size = torch.tensor([target["img_size"] for target in targets]).float()
        img_scale = torch.tensor([target["img_scale"] for target in targets]).float()

        annotations = {
            "bbox": boxes,
            "cls": labels,
            "img_size": img_size,
            "img_scale": img_scale,
        }

        return images, annotations, targets, image_ids


