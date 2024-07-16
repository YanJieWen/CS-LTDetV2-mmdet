# CS-LTDetV2-mmdetection-Implemention
<h2>
CS-LTDet: A Cross-Stage Feature Enhancement Detector For Large-Scale Train Crash Detection

<h2>
Bechmark: Crash2024

[![Dataset](https://img.shields.io/badge/GoogleDrive-Dataset-red)](https://drive.google.com/drive/folders/1BIhFc9dxOTIkJqC9OWZYQ1mbckGdIjrG?usp=drive_link)
<h2>


## Abstract 
Train crash experiments are one of the most direct and effective methods of developing crashworthiness and high-energy absorption trains, which is crucial for enhancing passive safety protection. High-Speed Camera (HSC), as a non-contact measurement method, can accurately capture the evolution of movement during crash process. However, manual annotation based on key points is proven to be tedious, costly, and error-prone. Moreover, simple feature informs is easily disturbed by the external environment. To this end, we propose a novel large-scale train crash intelligent detection network, named CS-LTDet. It aims to tackle precise location of objects and inter-class scale variance problems in datasets. Firstly, we equip the detector with a Cross-stage backbone Network (CSNet) that adopts dense connections to achieve multi-level feature fusion across intra-/inter-stages. Then, we introduce an Iterative Feature Pyramid Network (IFPN) inspired looking and thinking twice mechanism to pass FPN’s hierarchical feature maps back to the backbone for feature enhancement. These designs are intended to generate reliable proposals for further fine tuning in prediction head. Finally,  
extensive experiments are conducted on the first large-scale train collision dataset, Crash2024, which demonstrate that CS-LTDet achieves state-of-the-art performance compared to other detectors.  
Our code and dataset are made publicly available at [URL](https://github.com/YanJieWen/CS-LTDetV2-mmdet). 

## Contributions
[csnet](./mmdet/models/backbones/csnet.py)

## ACKNOWLEDGEMENT
[mmdetection](https://github.com/open-mmlab/mmdetection)
