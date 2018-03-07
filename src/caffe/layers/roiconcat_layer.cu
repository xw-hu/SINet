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
__global__ void ConcatForward(const int nthreads, const Dtype* bottom_normal, 
     const Dtype* bottom_small, const Dtype* bottom_hash, Dtype* top_data,
     const int channel, const int height, const int width) {
  CUDA_KERNEL_LOOP(index, nthreads) {

     int pw = index % width;
     int ph = (index / width) % height;
     int c = (index / width / height) % channel;
     int n = index / width / height / channel;

     bottom_hash += n*2;
     int new_n = ((bottom_hash[0] <0 )?0:bottom_hash[0]);
     
     //printf("[before]: %f\n",bottom_normal[((1*channel + c)*height + ph)*width + pw]);
     if (bottom_hash[1] == 1) //normal
     { 
        top_data[index] = bottom_normal[((new_n*channel + c)*height + ph)*width + pw];
     }
     else if (bottom_hash[1] == 2) //small
     {
        top_data[index] = bottom_small[((new_n*channel + c)*height + ph)*width + pw];
     }
  }
}

template <typename Dtype>
__global__ void ConcatBackward(const int nthreads, Dtype* bottom_normal, 
     Dtype* bottom_small, const Dtype* bottom_hash, const Dtype* top_diff,
     const int channel, const int height, const int width) {
  CUDA_KERNEL_LOOP(index, nthreads) {

     int pw = index % width;
     int ph = (index / width) % height;
     int c = (index / width / height) % channel;
     int n = index / width / height / channel;

     bottom_hash += n*2;
     int new_n = ((bottom_hash[0] <0 )?0:bottom_hash[0]);
     
     //printf("[before]: %f\n",bottom_normal[((1*channel + c)*height + ph)*width + pw]);
     if (bottom_hash[1] == 1) //normal
     { 
        bottom_normal[((new_n*channel + c)*height + ph)*width + pw] = top_diff[index];
     }
     else if (bottom_hash[1] == 2) //small
     {
        bottom_small[((new_n*channel + c)*height + ph)*width + pw] = top_diff[index];
     }
  }
}
/*
template <typename Dtype>
__global__ void Concat(const int nthreads, const Dtype* in_data,
    const bool forward, const int num_concats, const int concat_size,
    const int top_concat_axis, const int bottom_concat_axis,
    const int offset_concat_axis, Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int total_concat_size = concat_size * bottom_concat_axis;
    const int concat_num = index / total_concat_size;
    const int concat_index = index % total_concat_size;
    const int top_index = concat_index +
        (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
    if (forward) {
      out_data[top_index] = in_data[index];
    } else {
      out_data[index] = in_data[top_index];
    }
  }
}*/

template <typename Dtype>
void ROIConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //if (small_roi_ == 0 || normal_roi_ == 0) { return; }
  
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  const Dtype* bottom_normal = bottom[0]->gpu_data();
  const Dtype* bottom_small = bottom[1]->gpu_data();
  const Dtype* bottom_hash = bottom[3]->gpu_data();

  const int channel = bottom[0]->shape(1);
  const int height = bottom[0]->shape(2);
  const int width = bottom[0]->shape(3);

  ConcatForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_normal, bottom_small, bottom_hash, top_data,
        channel, height, width);
  CUDA_POST_KERNEL_CHECK;

  /*int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  const bool kForward = true;
  
  for (int i = 0; i < 2; ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    const int bottom_concat_size = bottom_concat_axis * concat_input_size_;
    const int nthreads = bottom_concat_size * num_concats_;
    Concat<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, bottom_data, kForward, num_concats_, concat_input_size_,
        top_concat_axis, bottom_concat_axis, offset_concat_axis, top_data);
    offset_concat_axis += bottom_concat_axis;
  }*/
}

template <typename Dtype>
void ROIConcatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->gpu_diff();
  int count = top[0]->count();
  Dtype* bottom_normal = bottom[0]->mutable_gpu_diff();
  Dtype* bottom_small = bottom[1]->mutable_gpu_diff();
  const Dtype* bottom_hash = bottom[3]->gpu_data();

  const int channel = top[0]->shape(1);
  const int height = top[0]->shape(2);
  const int width = top[0]->shape(3);  

  ConcatBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_normal, bottom_small, bottom_hash, top_diff,
        channel, height, width);
  CUDA_POST_KERNEL_CHECK;

  //if (small_roi_ == 0 || normal_roi_ == 0) { return; }
  /*const Dtype* top_diff = top[0]->gpu_diff();
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  const bool kForward = false;
  for (int i = 0; i < 2; ++i) {
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    if (propagate_down[i]) {
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      const int bottom_concat_size = bottom_concat_axis * concat_input_size_;
      const int nthreads = bottom_concat_size * num_concats_;
      Concat<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
          nthreads, top_diff, kForward, num_concats_, concat_input_size_,
          top_concat_axis, bottom_concat_axis, offset_concat_axis, bottom_diff);
    }
    offset_concat_axis += bottom_concat_axis;
  }*/
}

INSTANTIATE_LAYER_GPU_FUNCS(ROIConcatLayer);

}  // namespace caffe
