# SINet: A Scale-Insensitive Convolutional Neural Network for Fast Vehicle Detection

by Xiaowei Hu, Xuemiao Xu, Yongjie Xiao, Hao Chen, Shengfeng He, Jing Qin, and Pheng-Ann Heng

This implementation is written by Xiaowei Hu at the Chinese University of Hong Kong.

***

@article{hu2019sinet,        
&nbsp;&nbsp;&nbsp;&nbsp;  title={SINet: A Scale-Insensitive Convolutional Neural Network for Fast Vehicle Detection},        
&nbsp;&nbsp;&nbsp;&nbsp;  author={Hu, Xiaowei and Xu, Xuemiao and Xiao, Yongjie and Chen, Hao and He, Shengfeng and Qin, Jing and Heng, Pheng-Ann},        
&nbsp;&nbsp;&nbsp;&nbsp;  journal={IEEE Transactions on Intelligent Transportation Systems},        
&nbsp;&nbsp;&nbsp;&nbsp;  volume={20},        
&nbsp;&nbsp;&nbsp;&nbsp;  number={3},        
&nbsp;&nbsp;&nbsp;&nbsp;  pages={1010--1019},        
&nbsp;&nbsp;&nbsp;&nbsp;  year={2019},        
&nbsp;&nbsp;&nbsp;&nbsp;  publisher={IEEE}        
}

## LSVH Dataset

Our LSVH dataset is available for download at [Google Drive](https://drive.google.com/open?id=1raH0LF-hADB4BZmU9SAv19EWV6dxyQAw).       
The split of train.txt and test.txt is based on the Strategy 1, and please use `SINet/data/LSVH/strategy2.m` to generate the train.txt and test.txt based on the Strategy 2; see paper for details.


## Requirements

1. This code has been tested on Ubuntu 14.04, CUDA 7.0, cuDNN v3 with the NVIDIA TITAN X GPU and Ubuntu 16.04. CUDA 8.0 with the NVIDIA TITAN X(Pascal) GPU. 

2. We also need MATLAB scripts to run the auxiliary code, caffe MATLAB wrapper is required. Please build matcaffe before running the detection demo. 

3. cuDNN is required to avoid out-of-memory when training the models with VGG network.
  

## Installation
1. Clone the SINet repository, and we'll call the directory that you cloned SINet into `SINet`.

    ```shell
    git clone https://github.com/xw-hu/SINet.git
    ```

2. Build SINet (based on Caffe)
    
   Follow the Caffe installation instructions here: [http://caffe.berkeleyvision.org/installation.html](http://caffe.berkeleyvision.org/installation.html)   
   
   ```shell
   make all -j XX
   make matcaffe
   ```
   
 ## Training on KITTI car dataset

1. Download the KITTI dataset by yourself.

2. Enter the `SINet/models/PVA/` to download the PAVNet pretrained model:

   ```shell
   sh download_PVANet_imagenet.sh
   ```

3. Enter the `SINet/data/kitti/window_files`, and replace `/home/xwhu/KITTI/KITTI/` with your KITTI path.
  
   Another way is to run `mscnn_kitti_car_window_file.m` to generate the `txt` files that include the pathes of KITTI images.

4. Run `SINet/data/kitti/statistical_size.m` to calculate the parameters of `ROISplit` Layer in `trainval_2nd.prototxt`. 

5. (optional) Run `SINet/data/kitti/anchor_parameter.m` to calculate the anchors of `ImageGtData` layer. This is determined by K-means.

6. Enter the `SINet/examples/kitti_car/SINet-pva-576-2-branch`.
7. In the command window, run (around 1 hour on a single TITAN X):
   
   ```shell
   sh train_first_stage.sh
   ```
8. Use MATLAB to run `weight_2nd_ini.m`
9. In the command window, run (around 13.5 hours on a single TITAN X):

   ```shell
   sh train_second_stage.sh
   ```

Tip: If the training does not converge, try some other random seeds. You should obtain a fair performance after a few tries. Due to the randomness, you are difficult to fully reproduce the same models, but the performance should be close.


## Testing on KITTI car dataset

1. Use MATLAB to run `run_SINet_2_branch.m` in `SINet/examples/kitti_car`. It will generate the detection results in `SINet/examples/kitti_car/detections`. (In `run_SINet_detection.m`, let `show = 1`, we can show and save the detection results, but the speed is slower.)

2. We can get the quantitive results (average precision) in three levels: "easy", "moderate" and "hard" (same as the KITTI benchmark).

3. Without using cuDNN in testing, the running speed is higher.

## Training on other datasets

1. Enter the `SINet/data/kitti/` and modify the code `mscnn_kitti_car_window_file.m` to generate the `txt` files for your datasets.

2. Modify the parameters and the pathes of input images in `trainval_1st.prototxt` and  `trainval_2nd.prototxt`.

3. Others are the same as before.

## Testing on other datasets

1. Modify the `run_SINet_2_branch.m`, which generates the detection results in one `txt` file. 

2. Use the evaluation functions provided by KITTI or other benchmarks to calculate the quantitative results (in `SINet/examples/lsvh_result`, we use the VOC2011 evaluation code to calculate the mAP in our [LSVH dataset](https://drive.google.com/open?id=1yHeuZia3pbcbn8OLkotJGJGhczI7gM3e).

