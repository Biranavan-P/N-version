# N-Version System
This code implements an [N-Version](https://ieeexplore.ieee.org/document/8806018) System, an approach proposed by F. Machida and expanded with an [NXMI](https://link.springer.com/article/10.1007/s11623-023-1803-z) design by  Jass et al. The N-Version System is designed to enhance the reliability of machine learning models in safety-critical systems by employing model and input diversities.

The system includes three different models for image segmentation: VGG16-Unet, MobileNet-SegNet, and WCID (Where Can I Drive). The base code for VGG16-Unet and MobileNet-SegNet, modified for this project, can be found at: https://github.com/divamgupta/image-segmentation-keras (Last accessed: 01. 08. 2023).
The code is located under model/keras_segmentation and has been slightly modified). Meanwhile, the WCID model code, also slightly modified, is available at: https://github.com/FloHofstetter/wcid-keras (Last accessed: 01. 08. 2023).

The weights for these models are stored under their respective directories:
- VGG16-Unet: `vgg_unet/weights`
- MobileNet-SegNet: `mobilenet_segnet/weights`
- WCID: `model/wcid_keras/weights`

## Usage
The system requires an installation of Anaconda. Once Anaconda is installed, the following command in the project's root directory will install all necessary packages:

```bash
conda create -n n_version --file environment.txt
```

```bash
conda activate n_version
```
```bash
python3 model/n_version.py --image_path path/to/image --output_path path/to/save/results
```

Make sure to adjust the paths and commands based on your specific system setup and file locations.
