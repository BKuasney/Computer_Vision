
# OpenCV Installation Guide
 
This guide will show you how to install the latest version of OpenCV from source from Raspberry Pi platform. 

## Preparation for install  

First of all, try  ```sudo raspi-config``` and expand your file system.  
After this please reboot your pi. You just need to input ```sudo reboot``` on terminal.  
(If you are using USB memory instead of SD card, you cannot expand file system, but it's ok.  
Please go to next step right below.)  

Uninstall unneeded things.  
```shell
sudo apt-get purge wolfram-engine
sudo apt-get purge libreoffice*
sudo apt-get clean
sudo apt-get autoremove
```

## Install dependencies  
I omit libraries for python3 from original guide.  
```shell
sudo apt-get update && sudo apt-get upgrade  
sudo apt-get install build-essential cmake pkg-config  
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev  
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev  
sudo apt-get install libxvidcore-dev libx264-dev  
sudo apt-get install libgtk2.0-dev  
sudo apt-get install libcanberra-gtk*  
sudo apt-get install libatlas-base-dev gfortran  
sudo apt-get install python2.7-dev  
```

## Install OpenCV-4.0.0  

Different from original arcticle, I didn't install opencv-contrib.  
```shell
cd ~
wget -O opencv.zip https://github.com/opencv/opencv/archive/3.4.0.zip
unzip opencv.zip
rm opencv.zip
```
OK, you've downloaded OpenCV.
This directory's size is around 1.8GB.  
```shell
wget https://bootstrap.pypa.io/get-pip.py  
sudo python get-pip.py  
sudo rm -rf ~/.cache/pip  
pip install numpy  
```
Now, we will build it.  
```shell
cd ~/opencv-3.4.0/  
mkdir build  
cd build  
cmake -D CMAKE_BUILD_TYPE=RELEASE \  
    -D CMAKE_INSTALL_PREFIX=/usr/local \  
    -D ENABLE_NEON=ON \  
    -D ENABLE_VFPV3=ON \  
    -D BUILD_TESTS=OFF \  
    -D INSTALL_PYTHON_EXAMPLES=OFF \  
    -D BUILD_EXAMPLES=OFF ..  
```
And install.  
```shell    
make -j4  
```
This is the last commands for install.  
```shell
sudo make install
sudo ldconfig
```

## Check
Check whether OpenCV is installed correctly folling the command below.  

```
$ python  
>>> import cv2
>>> cv2.__version__
'4.0.0'
```

## Reference  
Optimizing OpenCV on the Raspberry Pi:  
https://www.pyimagesearch.com/2018/09/26/install-opencv-4-on-your-raspberry-pi/
