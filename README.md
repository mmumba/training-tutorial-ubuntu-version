# training-tutorial-ubuntu-version
Reproducing How To Train an "Object Detection Classifier Using TensorFlow-GPU 1.5 (GPU) on Windows 10" on ubuntu notes and observations.

(NOTE: Test out docker version of Tensorflow setup:https://www.tensorflow.org/install/ )

## Step 1: Install Tensorflow-GPU and Conda python 3.6

### Step 1a: Clone repository
```bash
git clone https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10.git
```

### Step 1b: Install upgrade tensorflow-gpu
 ```bash
    pip install --upgrade tensorflow-gpu
 ```
OUTPUT:
```
Successfully installed :
	* keras-applications-1.0.6 
	* keras-preprocessing-1.0.5
	* protobuf-3.6.1 
	* tensorboard-1.11.0
	* tensorflow-gpu-1.11.0
```
### Step 1c: Install CUDA v9.0 and cuDNN v.7.0

If already installed, check CUDA version

```bash
   nvcc --version
```
OUTPUT:
```
Cuda compilation tools, release 9.0, V9.0.176
```

check cuDNN version

```bash
   cat /usr/include/x86_64-linux-gnu/cudnn_v*.h | grep CUDNN_MAJOR -A 2
```
OUTPUT:
```
#define CUDNN_MAJOR 7
#define CUDNN_MINOR 0
#define CUDNN_PATCHLEVEL 3
```

## Step 1d: Install and upgrade Anaconda python 3.6

```bash
    conda install python=3.6
```



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


# Configuring Training.
Step to change fine-tune checkpoint should be line 106 (instead of 110 as given in original windows tutorial)

```bash
    gradient_clipping_by_norm: 10.0
  fine_tune_checkpoint: "/<YOUR-TENSORFLOW>/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"
```

same applies for following lines :
   Line 126 --> 123
   Line 128 --> 125
   Line 132 --> 129


 For Line 132 in Windows tutorial, "..\images\test" is used... this should be "..\images\train"


Incompatibility issue with python 3.5 when running training command 

  "Issue is with models/research/object_detection/utils/learning_schedules.py lines 167-169. Currently it is"

```bash
	rate_index = tf.reduce_max(tf.where(tf.greater_equal(global_step, boundaries),
                                      range(num_boundaries),
                                      [0] * num_boundaries))
```
Wrap list() around the range() like this:

```bash
rate_index = tf.reduce_max(tf.where(tf.greater_equal(global_step, boundaries),
                                     list(range(num_boundaries)),
                                      [0] * num_boundaries))
```
From models issues: 
```
 https://github.com/tensorflow/models/issues/3705#issuecomment-375563179
```
When accessing tensorboard, open new terminal and activate same tensorflow env you started your training in.

```bash
   source  activate tensorflow1
```


