# KDBBN
[![DOI](https://zenodo.org/badge/454431366.svg)](https://zenodo.org/badge/latestdoi/454431366)
## Overview
This repository is code of our paper *Machine vision-assisted identification of the lung adenocarcinoma category and high-risk tumor area based on CT images*.

## Datasets
The original datasets including metadata files used in the paper can be found in the website https://osf.io/5aqe4/.

## Codes

### 1. Transform the original CT datasets to the dataset used in the deep learning model in the paper.
The related codes are provided in the `./lung_code/trans_data` folder (`readDCM.py` shows the DCM files and distinguishes two different kinds, then with `crop4.py` we can change them into the same kind, finally with `trans2jpg.py` we transform dcm to jpg files). The datasets used for the deep learning model should be divided into 3 folders according to the adenocarcinoma categories.

### 2. Preprocessing stage
In `./lung_code/preprocessing` folder, `contour.py` is the SROI extractor, `background.py` changes the background of ROI into black or white, `crf.py` is the CRF extractor, `cut.py` is the Crop Background extractor and the $\mu$ and $\mu_t$ can be controlled by miu and miut. In addition, in `findROI.py` we provide several different ROI comparation. `mix.py` represents the part of the rebalancing unit and can generate the rebalance data.

### 3. KDBBN implementation and finetune
`./lung_code/Implement_concat.py` when TRAIN=True, it trains the whole framework, when TRAIN=False, it predicts. The weight can be finetuned or dynamically weighted. `./lung_code/cam.py` or `apply_gradcam.py` provides the codes of the heatmap generated through CAM or GRADCAM. `./lung_code/KD.py` shows the knowledge distillation procedure.

## Citation
If you want to use our data or codes for academic use, please cite our paper.

## License
The provided data and codes are strictly for academic purposes only. If you are interested in using our technology for any commercial use, please feel free to contact us

