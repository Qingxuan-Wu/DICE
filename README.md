# Introduction
Official code of the paper [DICE: End-to-end Deformation Capture of Hand-Face Interactions from a Single Image](https://arxiv.org/abs/2406.17988).

[【Project Page]](https://frank-zy-dou.github.io/projects/DICE/index.html) [[Paper]](https://arxiv.org/abs/2406.17988) [[Video]](https://www.youtube.com/watch?v=4ZuZveSElWE)

![Screenshot 2024-07-06 at 17 16 09](https://github.com/Qingxuan-Wu/DICE/assets/174913120/1c9494e3-d2f3-4f39-891d-dc0efcf49cd5)

Abstract: Reconstructing 3D hand-face interactions with deformations from a single image is a challenging yet crucial task with broad applications in AR, VR, and gaming. The challenges stem from self-occlusions during single-view hand-face interactions, diverse spatial relationships between hands and face, complex deformations, and the ambiguity of the single-view setting. The first and only method for hand-face interaction recovery, Decaf, introduces a global fitting optimization guided by contact and deformation estimation networks trained on studio-collected data with 3D annotations. However, Decaf suffers from a time-consuming optimization process and limited generalization capability due to its reliance on 3D annotations of hand-face interaction data. To address these issues, we present DICE, the first end-to-end method for Deformation-aware hand-face Interaction reCovEry from a single image. DICE estimates the poses of hands and faces, contacts, and deformations simultaneously using a Transformer-based architecture. It features disentangling the regression of local deformation fields and global mesh vertex locations into two network branches, enhancing deformation and contact estimation for precise and robust hand-face mesh recovery. To improve generalizability, we propose a weakly-supervised training approach that augments the training set using in-the-wild images without 3D ground-truth annotations, employing the depths of 2D keypoints estimated by off-the-shelf models and adversarial priors of poses for supervision. Our experiments demonstrate that DICE achieves state-of-the-art performance on a standard benchmark and in-the-wild data in terms of accuracy and physical plausibility. Additionally, our method operates at an interactive rate (20 fps) on an Nvidia 4090 GPU, whereas Decaf requires more than 15 seconds for a single image. Our code will be publicly available upon publication.

# Citation
```
@article{wu2024dice,
  title={DICE: End-to-end Deformation Capture of Hand-Face Interactions from a Single Image},
  author={Wu, Qingxuan and Dou, Zhiyang and Xu, Sirui and Shimada, Soshi and Wang, Chen and Yu, Zhengming and Liu, Yuan and Lin, Cheng and Cao, Zeyu and Komura, Taku and others},
  journal={arXiv preprint arXiv:2406.17988},
  year={2024}
}
```
