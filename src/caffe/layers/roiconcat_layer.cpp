// ------------------------------------------------------------------
// Copyright (c) 2016 
// The Chinese University of Hong Kong
// Written by Hu Xiaowei
// ------------------------------------------------------------------

#include <vector>
#include <iostream>

#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"

using namespace std;

namespace caffe {

template <typename Dtype>
void ROIConcatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void ROIConcatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  small_roi_ = static_cast<int>((bottom[2]->cpu_data())[0]);
  normal_roi_ = static_cast<int>((bottom[2]->cpu_data())[1]);
  //cout<<"small_roi_: "<<small_roi_<<" normal_roi_: "<<normal_roi_<<endl;

  if (small_roi_ == 0) { //just pass the normal roi bottom
    top[0]->Reshape(bottom[0]->shape());
    return;
  }
  if (normal_roi_ == 0 ) { //just pass the small roi bottom 
    top[0]->Reshape(bottom[1]->shape());
    return;
  }
  
  concat_axis_ = 0; //param axis:0

  // Initialize find the shape 
  int roi_num = bottom[0]->shape(0)+bottom[1]->shape(0);
  CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1));
  CHECK_EQ(bottom[0]->shape(2), bottom[1]->shape(2));
  CHECK_EQ(bottom[0]->shape(3), bottom[1]->shape(3));

  num_concats_ = bottom[1]->count(0, concat_axis_); //=1

  concat_input_size_ = bottom[1]->count(concat_axis_ + 1); //=c*w*h

  //cout<<"concat_input_size_: "<<concat_input_size_<<endl;

  top[0]->Reshape(roi_num, bottom[1]->shape(1), bottom[1]->shape(2), bottom[1]->shape(3));
}

template <typename Dtype>
void ROIConcatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  Dtype* top_data = top[0]->mutable_cpu_data();

  const Dtype* bottom_hash_bool = bottom[3]->cpu_data();
  //int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_); //total number of ROI

/////////////////////////////////////////////////////////////////cuhk
  const Dtype* bottom_normal = bottom[0]->cpu_data();
  const Dtype* bottom_small = bottom[1]->cpu_data();

  Blob<Dtype> bottom_cpy; //copy normal data
  bottom_cpy.Reshape(bottom[1]->shape());
  Dtype* bottom_cpy_data = bottom_cpy.mutable_cpu_data();

  caffe_copy(bottom[1]->shape(0)*concat_input_size_,bottom_small,bottom_cpy_data);

  int normal_count = 0;
  int small_count = 0;

  for (int i=0; i<top_concat_axis; i++) //for every roi 
  {
     int index = bottom_hash_bool[2*i+1]; //1 normal   2 small

     if (index == 2 )
     {
         caffe_copy(concat_input_size_, //num_of_ROI * [c*w*h] total number of value
          bottom_small + small_count * concat_input_size_, //bottom -> top
          top_data + i * concat_input_size_);

        small_count++;
     
     }
     else //if (index == 1)
     {
        caffe_copy(concat_input_size_, //num_of_ROI * [c*w*h] total number of value
          bottom_normal + normal_count * concat_input_size_, //bottom -> top
          top_data + i * concat_input_size_);
        
        normal_count++;

     }
  }

  /*for (int i = 0; i < 2; ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_); //num of (ROI) concat_axis for each bottom  

    cout<<"bottom_concat_axis: "<<bottom_concat_axis<<endl;

    for (int n = 0; n < num_concats_; ++n) {
      caffe_copy(bottom_concat_axis * concat_input_size_, //num_of_ROI * [c*w*h] total number of value
          bottom_data + n * bottom_concat_axis * concat_input_size_, //bottom -> top
          top_data + (n * top_concat_axis + offset_concat_axis)
              * concat_input_size_);
    }
    offset_concat_axis += bottom_concat_axis;
  }*/
}

template <typename Dtype>
void ROIConcatLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      NOT_IMPLEMENTED;
  /*
  const Dtype* top_diff = top[0]->cpu_diff();
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  for (int i = 0; i < 2; ++i) {
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    if (propagate_down[i]) {
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
      for (int n = 0; n < num_concats_; ++n) {
        caffe_copy(bottom_concat_axis * concat_input_size_, top_diff +
            (n * top_concat_axis + offset_concat_axis) * concat_input_size_,
            bottom_diff + n * bottom_concat_axis * concat_input_size_);
      }
    }
    offset_concat_axis += bottom_concat_axis;
  }*/
}

#ifdef CPU_ONLY
STUB_GPU(ROIConcatLayer);
#endif

INSTANTIATE_CLASS(ROIConcatLayer);
REGISTER_LAYER_CLASS(ROIConcat);

}  // namespace caffe
