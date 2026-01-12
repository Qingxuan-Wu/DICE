## As a first step, the video data need to be processed by the script below.
## The script creates the cropped image and segmentation masks under the path specified by --save_path.
## In the example below, the DecafDataset is located at /mnt/d/DecafDataset/, and the generated images will be saved under /mnt/d/DecafDataset_images/

python image_cropper.py  --data_path /mnt/d/DecafDataset/  --save_path /mnt/d/DecafDataset_images/


## Run the script below to show an example dataloading. The script will show fetched images after applying augmentations (normalization, rotation and adding random background).
## --image_data_path should specify the image dataset generated in the previous step.

python simple_loading_script.py --data_path /mnt/d/DecafDataset/  --image_data_path /mnt/d/DecafDataset_images/


