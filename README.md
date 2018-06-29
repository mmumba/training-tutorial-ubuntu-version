# training-tutorial-ubuntu-version
Reproducing How To Train an "Object Detection Classifier Using TensorFlow 1.5 (GPU) on Windows 10" on ubuntu notes and observations.




## Setting the python path for conda in Ubuntu

```bash 
   export PYTHONPATH=/home/sink/obj-detect-ws/tensorflow1/models:/home/sink/obj-detect-ws/tensorflow1/models/research:/home/sink/obj-detect-ws/tensorflow1/models/research/slim
```


## installing LabelImg
 Using
  https://mlnotesblog.wordpress.com/2017/12/16/how-to-install-labelimg-in-ubuntu-16-04

```bash
  git clone https://github.com/tzutalin/labelImg
  cd labelImg/
  make qt5py3 
  python labelImg.py 
```
original website: 
https://github.com/tzutalin/labelImg


Best practiced for splitting data for training, deve and testing. 

https://cs230-stanford.github.io/train-dev-test-split.html
