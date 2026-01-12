# DICE

## Introduction
Official code of the paper [DICE: End-to-end Deformation Capture of Hand-Face Interactions from a Single Image](https://arxiv.org/abs/2406.17988).

[[Project Page]](https://frank-zy-dou.github.io/projects/DICE/index.html) [[Paper]](https://arxiv.org/abs/2406.17988) [[Video]](https://www.youtube.com/watch?v=4ZuZveSElWE)

![Screenshot 2024-07-06 at 17 16 09](https://github.com/Qingxuan-Wu/DICE/assets/174913120/1c9494e3-d2f3-4f39-891d-dc0efcf49cd5)

Abstract: Reconstructing 3D hand-face interactions with deformations from a single image is a challenging yet crucial task with broad applications in AR, VR, and gaming. The challenges stem from self-occlusions during single-view hand-face interactions, diverse spatial relationships between hands and face, complex deformations, and the ambiguity of the single-view setting. The first and only method for hand-face interaction recovery, Decaf, introduces a global fitting optimization guided by contact and deformation estimation networks trained on studio-collected data with 3D annotations. However, Decaf suffers from a time-consuming optimization process and limited generalization capability due to its reliance on 3D annotations of hand-face interaction data. To address these issues, we present DICE, the first end-to-end method for Deformation-aware hand-face Interaction reCovEry from a single image. DICE estimates the poses of hands and faces, contacts, and deformations simultaneously using a Transformer-based architecture. It features disentangling the regression of local deformation fields and global mesh vertex locations into two network branches, enhancing deformation and contact estimation for precise and robust hand-face mesh recovery. To improve generalizability, we propose a weakly-supervised training approach that augments the training set using in-the-wild images without 3D ground-truth annotations, employing the depths of 2D keypoints estimated by off-the-shelf models and adversarial priors of poses for supervision. Our experiments demonstrate that DICE achieves state-of-the-art performance on a standard benchmark and in-the-wild data in terms of accuracy and physical plausibility. Additionally, our method operates at an interactive rate (20 fps) on an Nvidia 4090 GPU, whereas Decaf requires more than 15 seconds for a single image. Our code will be publicly available upon publication.


## Inference

### Environment Preparation

Create Conda environment: 
```
conda create -n dice python=3.9
conda activate dice
```

Install required packages:
```
pip install -r requirements.txt
```

Install `manopth`: 
```
git clone https://github.com/hassony2/manopth.git && cd manopth && git checkout 4f1dcad && pip install -e .  && cd ..
```

Install `pytorch3d`:
```
git clone https://github.com/facebookresearch/pytorch3d.git&&cd ./pytorch3d&&git checkout tags/v0.7.2&&pip install -e .&&cd ..
```

Install `apex` following [METRO](https://github.com/microsoft/MeshTransformer/blob/cefc4af50ca0fa4f58e677ad1836ed09a64b4d9a/docs/INSTALL.md).

### Dependency Files
- run `sh download_models.sh` in the root folder to download the pretrained HRNet-W64 checkpoint.
- create the folder `src/common/utils/human_model_files` and download the relevant files according to this [instruction](https://github.com/IDEA-Research/OSX?tab=readme-ov-file#4-directory).
- Download `head_mesh_transforms.pt` and `hand_mesh_transforms.pt` [here](https://drive.google.com/drive/folders/1go919S1HxzJ_6Mvgsx1-QsOwWNiAYe8L?usp=sharing) and save to the root folder.
- Download `head_ref_vs.pt`, `rh_ref_vs.pt`, and `stiffness_final.npy` [here](https://drive.google.com/drive/folders/1go919S1HxzJ_6Mvgsx1-QsOwWNiAYe8L?usp=sharing) and place it in `src/modeling/data/`.
- Download `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from [SMPLify](http://smplify.is.tue.mpg.de/), and place it in `src/modeling/data`.
- Download `MANO_RIGHT.pkl` from [MANO](https://mano.is.tue.mpg.de/), and place it in `src/modeling/data`.
- Download `model.bin` from [here](https://drive.google.com/drive/folders/1go919S1HxzJ_6Mvgsx1-QsOwWNiAYe8L?usp=sharing) and place it in `checkpoints`.

### Usage

To run inference, use our script: `sh infer.sh` to run inference on sample images from `assets/images`. Visualizations and output meshes are saved to `output/example_inference`.

For best results, crop input images to put the head and the face near center before running inference.

## Citation
```
@inproceedings{wudice,
  title={DICE: End-to-end Deformation Capture of Hand-Face Interactions from a Single Image},
  author={Wu, Qingxuan and Dou, Zhiyang and Xu, Sirui and Shimada, Soshi and Wang, Chen and Yu, Zhengming and Liu, Yuan and Lin, Cheng and Cao, Zeyu and Komura, Taku and others},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```

## Acknowledgements
Our implementation and experiments are built on top of open-source GitHub repositories. We thank all the authors who made their code public, which tremendously accelerates our project progress. 

[HRNet/HRNet-Image-Classification](https://github.com/HRNet/HRNet-Image-Classification)

[huggingface/transformers](https://github.com/huggingface/transformers)

[microsoft/MeshTransformer](https://github.com/microsoft/MeshTransformer)

[MPI-IS/mesh](https://github.com/MPI-IS/mesh)

[facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d)