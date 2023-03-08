from pytorch_lightning import Trainer
from pathlib import Path
from utils.loaders import PigsDatasetAdaptor,EfficientDetDataModule
from model import EfficientDetModel
import torch

import argparse

parser = argparse.ArgumentParser(description="Train Object detector EfficientDet")
parser.add_argument("--data-dir",default='data',type=str)
parser.add_argument("--no-cuda",action="store_true")
parser.add_argument("--gpu-id",default=0,type=int)
args = parser.parse_args()



def main():
    def_img_size = 512
    CLASSES = 1
    DEF_IMG_SIZE = 512
    ARCHITECTURE = "tf_efficientdet_d0"   
    BATCH_SIZE = 4
    WORKERS = 8
    MAX_EPOCHS = 15
    IOU_THRESH = 0.2
    SIGMA = 0.2
    model_architecture = "tf_efficientdet_d0"
    dataDir=Path(r'C:\Users\USER\NTNU Lab Works\svin_detection\clean_data\Norsvin_New\Norsvin\train\images')
    annFiles = Path(r'C:\Users\USER\NTNU Lab Works\svin_detection\clean_data\Norsvin_New\Norsvin\train\annotations')
    pig_train_ds = PigsDatasetAdaptor(dataDir,annFiles)
    valid_dataDir=Path(r'C:\Users\USER\NTNU Lab Works\svin_detection\clean_data\Norsvin_New\Norsvin\val\images')
    test_annFiles = Path(r'C:\Users\USER\NTNU Lab Works\svin_detection\clean_data\Norsvin_New\Norsvin\val\annotations')
    pig_valid_ds = PigsDatasetAdaptor(valid_dataDir,test_annFiles)
    dm = EfficientDetDataModule(train_dataset_adaptor=pig_train_ds, validation_dataset_adaptor=pig_valid_ds,num_workers=WORKERS,batch_size=BATCH_SIZE)
    model = EfficientDetModel(None,num_classes=CLASSES,img_size=DEF_IMG_SIZE,wbf_iou_threshold=IOU_THRESH,sigma=SIGMA,model_architecture=ARCHITECTURE)
    trainer = Trainer(max_epochs=MAX_EPOCHS, num_sanity_val_steps=1,accelerator="gpu",devices = [0])
    torch.cuda.empty_cache()
    trainer.fit(model, dm)
    torch.save(model.state_dict(), f'trained_outputs/trained_weights{model_architecture}_{def_img_size}_test_without_barlow')
    print("saving model......")
    dest_file = f'trained_outputs/trained_weights_{model_architecture}_{def_img_size}'
    print("Model has been saved to  ",dest_file)
    torch.save(model.state_dict(), f'trained_outputs/trained_weights_{model_architecture}_{def_img_size}')
    outputs = trainer.validate(model=model,datamodule=dm)
    #test_outputs = trainer.test(model=model,datamodule=dm)
    
def main_2():
    def_img_size = 512
    CLASSES = 1
    DEF_IMG_SIZE = 512
    ARCHITECTURE = "tf_efficientdet_d0"   
    BATCH_SIZE = 4
    WORKERS = 8
    MAX_EPOCHS = 15
    IOU_THRESH = 0.2
    SIGMA = 0.2
    model_architecture = "tf_efficientdet_d0"
    saved_model = r"C:\Users\USER\NTNU Lab Works\self_supervised_learning\model_save\trained_weights_tf_efficientdet_d0_512"

    dataDir=Path(r'C:\Users\USER\NTNU Lab Works\svin_detection\clean_data\Norsvin_New\Norsvin\train\images')
    annFiles = Path(r'C:\Users\USER\NTNU Lab Works\svin_detection\clean_data\Norsvin_New\Norsvin\train\annotations')
    valid_dataDir=Path(r'C:\Users\USER\NTNU Lab Works\svin_detection\clean_data\Norsvin_New\Norsvin\val\images')
    test_annFiles = Path(r'C:\Users\USER\NTNU Lab Works\svin_detection\clean_data\Norsvin_New\Norsvin\val\annotations')

    pig_train_ds = PigsDatasetAdaptor(dataDir,annFiles)
    pig_valid_ds = PigsDatasetAdaptor(valid_dataDir,test_annFiles)

    dm = EfficientDetDataModule(train_dataset_adaptor=pig_train_ds, validation_dataset_adaptor=pig_valid_ds,num_workers=WORKERS,batch_size=BATCH_SIZE)
    model = EfficientDetModel(saved_model,num_classes=CLASSES,img_size=DEF_IMG_SIZE,wbf_iou_threshold=IOU_THRESH,sigma=SIGMA,model_architecture=ARCHITECTURE)
    trainer = Trainer(max_epochs=MAX_EPOCHS, num_sanity_val_steps=1,accelerator="gpu",devices = [0])
    trainer.fit(model, dm)
    print("Finished Training :.....")
    torch.save(model.state_dict(), f'final_train/trained_weights{model_architecture}_{def_img_size}_test_2')
    print("saving model......")
    dest_file = f'final_train/trained_weights{model_architecture}_{def_img_size}_test'
    print("Model has been saved to {dest_file}")
    outputs = trainer.validate(model=model,datamodule=dm)



if __name__ == "__main__":
    main()
