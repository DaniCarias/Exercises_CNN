import os
import shutil

def cut_dataset():
    TRAIN_IMG_PATH = "../../data/forest_seg/train/imgs"
    TRAIN_MASK_PATH = "../../data/forest_seg/train/masks"
    VALI_IMG_PATH = "../../data/forest_seg/validation/imgs"
    VALI_MASK_PATH = "../../data/forest_seg/validation/masks"
    
    NEW_TRAIN_IMG_PATH = "../../data/forest_seg/half_size/train/imgs"
    NEW_TRAIN_MASK_PATH = "../../data/forest_seg/half_size/train/masks"
    NEW_VALI_IMG_PATH = "../../data/forest_seg/half_size/validation/imgs"
    NEW_VALI_MASK_PATH = "../../data/forest_seg/half_size/validation/masks"

   # Get a list of all images in the directory
    train_img_files = sorted(os.listdir(TRAIN_IMG_PATH))
    train_mask_files = sorted(os.listdir(TRAIN_MASK_PATH))
    validation_img_files = sorted(os.listdir(VALI_IMG_PATH))
    validation_mask_files = sorted(os.listdir(VALI_MASK_PATH))

    # Calculate the index of the middle image
    train_middle_index = len(train_img_files) // 2
    val_middle_index = len(validation_img_files) // 2

    # Copy the first half of the images and masks to the new directories
    for i in range(train_middle_index):
        shutil.copy(os.path.join(TRAIN_IMG_PATH, train_img_files[i]), NEW_TRAIN_IMG_PATH)
        shutil.copy(os.path.join(TRAIN_MASK_PATH, train_mask_files[i]), NEW_TRAIN_MASK_PATH)
    
    for i in range(val_middle_index):
        shutil.copy(os.path.join(VALI_IMG_PATH, validation_img_files[i]), NEW_VALI_IMG_PATH)
        shutil.copy(os.path.join(VALI_MASK_PATH, validation_mask_files[i]), NEW_VALI_MASK_PATH)

if __name__ == "__main__":
    cut_dataset()