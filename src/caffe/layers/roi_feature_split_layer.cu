// ------------------------------------------------------------------
// Copyright (c) 2016 
// The Chinese University of Hong Kong
// Written by Hu Xiaowei
// ------------------------------------------------------------------

#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void FeatureSplitForward(const int nthreads, const Dtype* bottom_data, 
     const Dtype* bottom_hash, Dtype* top_small_data, Dtype* top_middle_data, Dtype* top_large_data, const int channel, const int height, const int width, const int branch_num) {
  CUDA_KERNEL_LOOP(index, nthreads) {

     int pw = index % width;
     int ph = (index / width) % height;
     int c = (index / width / height) % channel;
     int n = index / width / height / channel;

     bottom_hash += n*2;
     int new_n = ((bottom_hash[0] <0 )?0:bottom_hash[0]);
     
     if (branch_num == 3) 
     {
        if (bottom_hash[1] == 1) //small
        { 
           top_small_data[((new_n*channel + c)*height + ph)*width + pw] = bottom_data[index];
        }
        else if (bottom_hash[1] == 2) //middle
        {
           top_middle_data[((new_n*channel + c)*height + ph)*width + pw] = bottom_data[index];
        }
        else if (bottom_hash[1] == 3) //large
        {
           top_large_data[((new_n*channel + c)*height + ph)*width + pw] = bottom_data[index];
        }
     }
     else 
     {
        if (bottom_hash[1] == 1) //small
        { 
           top_small_data[((new_n*channel + c)*height + ph)*width + pw] = bottom_data[index];
        }
        else if (bottom_hash[1] == 2) //large
        {
           top_large_data[((new_n*channel + c)*height + ph)*width + pw] = bottom_data[index];
        }
     }
  }
}

template <typename Dtype>
__global__ void FeatureSplitBackward(const int nthreads, Dtype* bottom_diff, 
     const Dtype* bottom_hash, const Dtype* top_small_diff, const Dtype* top_middle_diff, const Dtype* top_large_diff, const int channel, const int height, const int width, const int branch_num) {
  CUDA_KERNEL_LOOP(index, nthreads) {

     int pw = index % width;
     int ph = (index / width) % height;
     int c = (index / width / height) % channel;
     int n = index / width / height / channel;

     bottom_hash += n*2;
     int new_n = ((bottom_hash[0] <0 )?0:bottom_hash[0]);
     
     
     if (branch_num == 3) 
     {
        if (bottom_hash[1] == 1) //small
        {  
           bottom_diff[index]=top_small_diff[((new_n*channel + c)*height + ph)*width + pw];
        }
        else if (bottom_hash[1] == 2) //middle
        {
           bottom_diff[index]=top_middle_diff[((new_n*channel + c)*height + ph)*width + pw];
        }
        else if (bottom_hash[1] == 3) //large
        {
           bottom_diff[index]=top_large_diff[((new_n*channel + c)*height + ph)*width + pw];
        }
     }
     else
     {
        if (bottom_hash[1] == 1) //small
        {  
           bottom_diff[index]=top_small_diff[((new_n*channel + c)*height + ph)*width + pw];
        }
        else if (bottom_hash[1] == 2) //large
        {
           bottom_diff[index]=top_large_diff[((new_n*channel + c)*height + ph)*width + pw];
        }
     }
  }
}

template <typename Dtype>
void ROIFeatureSplitLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  Dtype* top_large_data = NULL;
  Dtype* top_middle_data = NULL;
  Dtype* top_small_data = NULL;

  if (branch_num_ == 3)
  {
     top_small_data = top[0]->mutable_gpu_data();
     top_middle_data = top[1]->mutable_gpu_data();
     top_large_data = top[2]->mutable_gpu_data();
  }
  else 
  {
     top_small_data = top[0]->mutable_gpu_data();
     top_large_data = top[1]->mutable_gpu_data();
  }

  int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_hash = bottom[2]->gpu_data();

  FeatureSplitForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom_hash, top_small_data, top_middle_data, top_large_data, channels_, height_, width_, branch_num_);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void ROIFeatureSplitLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_large_diff;
  const Dtype* top_middle_diff;
  const Dtype* top_small_diff;

  if (branch_num_ == 3)
  {
     top_small_diff = top[0]->gpu_diff();
     top_middle_diff = top[1]->gpu_diff();
     top_large_diff = top[2]->gpu_diff();
  }
  else 
  {
     top_small_diff = top[0]->gpu_diff();
     top_large_diff = top[1]->gpu_diff();
  }

  int count = bottom[0]->count();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* bottom_hash = bottom[2]->gpu_data();

  FeatureSplitBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_diff, bottom_hash, top_small_diff, top_middle_diff, top_large_diff, channels_, height_, width_,branch_num_);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(ROIFeatureSplitLayer);

}  // namespace caffe
