# training-tutorial-ubuntu-version
Reproducing How To Train an "Object Detection Classifier Using TensorFlow 1.5 (GPU) on Windows 10" on ubuntu notes and observations.




## Setting the python path for conda in Ubuntu

```bash 
   export PYTHONPATH=/home/sink/obj-detect-ws/tensorflow1/models:/home/sink/obj-detect-ws/tensorflow1/models/research:/home/sink/obj-detect-ws/tensorflow1/models/research/slim
```


## installing LabelImg

```bash
  git clone https://github.com/tzutalin/labelImg
  cd labelImg/
  make qt5py3 
  python labelImg.py 
```
original website: 
https://github.com/tzutalin/labelImg



