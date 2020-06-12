# Take the Scenic Route
Implemention of [Take the Scenic Route: Improving Generalization in Vision-and-Language Navigation](https://arxiv.org/abs/2003.14269)

## Initialize the repo
```
git clone --recursive VLNActionPriors.git
```

## get pybind as submodule
```
git submodule add -b master git@github.com:pybind/pybind11.git
```

## Create environment for installation
```
conda env create -f environment.yml

source activate AcP
```
## Install pytorch:
```
conda install pytorch=1.0.0 cuda100 -c pytorch
```
## Compile the Matterport3D environment
```
mkdir build && cd build
cmake ..
make
```
## Dataset Download

### R2R data:
```
bash download_r2r.sh
```
### Pre-computed image features:
```
mkdir -p img_features/
cd img_features/
wget https://www.dropbox.com/s/o57kxh2mn5rkx4o/ResNet-152-imagenet.zip?dl=1 -O ResNet-152-imagenet.zip
unzip ResNet-152-imagenet.zip
cd ..
```
### Download pre-trained speaker model (following Speaker Follower repo) 
```
bash download_speaker_release.sh
```
### Train the model/agent using Random Walk Data Augmentation
```
bash run_w_params.sh
```
