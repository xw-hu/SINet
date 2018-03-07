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
void ROIFeatureSplitLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ROIFeatureSplitParameter roi_feature_split_param = this->layer_param_.roi_feature_split_param();

  branch_num_ = roi_feature_split_param.branch_num();

  CHECK_GT(branch_num_, 1)
      << "split_area must be > 1";
  CHECK_LT(branch_num_, 4)
      << "split_area must be < 4";

  
}

template <typename Dtype>
void ROIFeatureSplitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  
  int top_small,top_middle,top_large;

  if (branch_num_ == 3)
  {
     small_roi_ = static_cast<int>((bottom[1]->cpu_data())[0]);
     middle_roi_ = static_cast<int>((bottom[1]->cpu_data())[1]);
     large_roi_ = static_cast<int>((bottom[1]->cpu_data())[2]);
     //cout<<"small_roi_: "<<small_roi_<<" normal_roi_: "<<normal_roi_<<endl;
 
     top_small = small_roi_;
     top_middle = middle_roi_;
     top_large = large_roi_;

     if (small_roi_ == 0) { //just pass the normal roi 
       top_small = 1; 
     }
     if (middle_roi_ == 0 ) { //just pass the small roi
       top_middle = 1;
     }
     if (large_roi_ == 0 ) { //just pass the small roi
       top_large = 1;
     }

     top[0]->Reshape(top_small, channels_, height_, width_);

     top[1]->Reshape(top_middle, channels_, height_, width_);

     top[2]->Reshape(top_large, channels_, height_, width_);
  }
  else
  {
     small_roi_ = static_cast<int>((bottom[1]->cpu_data())[0]);
     large_roi_ = static_cast<int>((bottom[1]->cpu_data())[1]);
 
     top_small = small_roi_;
     top_large = large_roi_;

     if (small_roi_ == 0) { //just pass the normal roi 
       top_small = 1; 
     }
     if (large_roi_ == 0 ) { //just pass the small roi
       top_large = 1;
     }

     top[0]->Reshape(top_small, channels_, height_, width_);

     top[1]->Reshape(top_large, channels_, height_, width_);
  }
}

template <typename Dtype>
void ROIFeatureSplitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      NOT_IMPLEMENTED;
}

template <typename Dtype>
void ROIFeatureSplitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(ROIFeatureSplitLayer);
#endif

INSTANTIATE_CLASS(ROIFeatureSplitLayer);
REGISTER_LAYER_CLASS(ROIFeatureSplit);

}  // namespace caffe
