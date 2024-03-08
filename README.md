# N-version
Dieser Code ist eine Implementation eines [N-Version-Systems](https://ieeexplore.ieee.org/document/8806018) nach F. Machida, welches durch ein [NXMI Ansatz](https://link.springer.com/article/10.1007/s11623-023-1803-z) von Jass et al. erweitert wurde. Die Implementation erfolgte durch Biranavan Parameswaran.

Das System besteht aus: VGG16-Unet, MobileNet-SegNet und [WCID](https://arxiv.org/abs/2004.07639).

Der Code für das VGG16-Unet und das MobileNet-SegNet wurde von Divam Gupta unter folgendem Link zur Verfügung gestellt: https://github.com/divamgupta/image-segmentation-keras (Zuletzt abgerufen am: 01. 08. 2023)
Der Code ist zu finden unter model/keras_segmentation und wurde leicht modifiziert.

Der Code für das WCID wurde von Florian Hofstetter unter folgendem Link zur Verfügung gestellt: https://github.com/FloHofstetter/wcid-keras (Zuletzt abgerufen am: 01. 08. 2023)
Der Code ist zu finden unter model/wcid_keras und wurde leicht modifiziert.

Die Gewichte sind unter folgendem Pfaden zu finden:
VGG16-Unet: vgg_unet/weights
MobileNet-SegNet: mobilenet_segnet/weights
WCID: model/wcid_keras/weight

## Nutzung
Es sollte eine [Anaconda](https://anaconda.org/) installiert sein. Im aktuellen Pfad des Projektes wird folgender Befehl alle wichtigen Pakete installieren:
```bash
conda create -n n_version --file environment.txt
```
Daraufhin muss die Umgebung aktiviert werden:
```
conda activate n_version
```
Für die Nutzung des N-Version-Systems muss folgender Befehl ausgeführt werden: 

```bash
python3 model/n_version.py --image_path path/to/image  --output_pth path/to/save/results
```
