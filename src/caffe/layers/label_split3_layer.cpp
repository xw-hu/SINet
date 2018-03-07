// ------------------------------------------------------------------
// Copyright (c) 2017
// The Chinese University of Hong Kong
// Written by Hu Xiaowei
// ------------------------------------------------------------------
// Use for three branches 

#include <vector>
#include <iostream>

#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"

using namespace std;

namespace caffe {

template <typename Dtype>
void LabelSplit3Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void LabelSplit3Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  small_roi_ = static_cast<int>((bottom[0]->cpu_data())[0]);
  middle_roi_ = static_cast<int>((bottom[0]->cpu_data())[1]);
  large_roi_ = static_cast<int>((bottom[0]->cpu_data())[2]);
  
  roi_num_ = small_roi_ + middle_roi_ + large_roi_;

  small_roi_ = (small_roi_==0)?1:small_roi_;
  middle_roi_ = (middle_roi_==0)?1:middle_roi_;
  large_roi_ = (large_roi_==0)?1:large_roi_;

  for (int i=0;i<4;i++) //small
  {
      top[i]->Reshape(small_roi_, bottom[i+2]->shape(1), bottom[i+2]->shape(2), bottom[i+2]->shape(3));
  }

  for (int i=4;i<8;i++) //middle
  {
      top[i]->Reshape(middle_roi_, bottom[i-2]->shape(1), bottom[i-2]->shape(2), bottom[i-2]->shape(3));
  }

  for (int i=8;i<12;i++) //large
  {
      top[i]->Reshape(large_roi_, bottom[i-6]->shape(1), bottom[i-6]->shape(2), bottom[i-6]->shape(3));
  }
}

template <typename Dtype>
void LabelSplit3Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_hash_bool = bottom[1]->cpu_data();  

  Dtype* top_data[12];
  const Dtype* bottom_data[4];

  for (int i=0; i<12; i++)
  {
      top_data[i] = top[i]->mutable_cpu_data();
  }
  for (int i=2; i<6; i++)
  {
      bottom_data[i-2] = bottom[i]->cpu_data();
  }

  for (int i=0; i<roi_num_; i++)
  {
      //bool small_or_not = (bottom_hash_bool[2*i+1]==static_cast<Dtype>(2));
      int roi_new_count = bottom_hash_bool[2*i]; //the number of rois for each class
      if(bottom_hash_bool[2*i+1]==static_cast<Dtype>(1)) //small
      {   
         for(int j=0; j<4; j++)
         {
            int channel_num = (bottom[j+2])->count(1); //c*w*h
            
            caffe_copy(channel_num, bottom_data[j]+i*channel_num, top_data[j]+roi_new_count*channel_num);
         }
      }
      else if(bottom_hash_bool[2*i+1]==static_cast<Dtype>(2)) //middle
      {
         for(int j=4; j<8; j++)
         {
            int channel_num = bottom[j-2]->count(1); //c*w*h
            
            caffe_copy(channel_num, bottom_data[j-4]+i*channel_num, top_data[j]+roi_new_count*channel_num);
         }
      }
      else //large
      {
         for(int j=8; j<12; j++)
         {
            int channel_num = bottom[j-6]->count(1); //c*w*h
            
            caffe_copy(channel_num, bottom_data[j-8]+i*channel_num, top_data[j]+roi_new_count*channel_num);
         }
      }
  }
}

#ifdef CPU_ONLY
STUB_GPU(LabelSplit3Layer);
#endif

INSTANTIATE_CLASS(LabelSplit3Layer);
REGISTER_LAYER_CLASS(LabelSplit3);

}  // namespace caffe
