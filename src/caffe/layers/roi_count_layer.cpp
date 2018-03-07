// ------------------------------------------------------------------
// Copyright (c) 2016 
// The Chinese University of Hong Kong
// Written by Hu Xiaowei
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

//#include <iostream>

using std::max;
using std::min;
using std::floor;
using std::ceil;
//using std::cout;
//using std::endl;

namespace caffe {

template <typename Dtype>
void ROICountLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ROICountParameter roi_count_param = this->layer_param_.roi_count_param();
  CHECK_GT(roi_count_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(roi_count_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = roi_count_param.pooled_h();
  pooled_width_ = roi_count_param.pooled_w();
  spatial_scale_ = roi_count_param.spatial_scale();
  pad_ratio_ = roi_count_param.pad_ratio();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void ROICountLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(2, 1, 1, 1);
  top[1]->Reshape(bottom[1]->num(),2,1,1); //hash function for different size of ROI (num for each class, index for normal[1] small[2])
  //top[2]->Reshape(bottom[1]->num(),1,1,1); //hash function for different size of ROI

}

template <typename Dtype>
void ROICountLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_rois = bottom[1]->cpu_data();
  // Number of ROIs
  int num_rois = bottom[1]->num();

  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* hash_table = top[1]->mutable_cpu_data();
  //Dtype* hash_bool_table = top[2]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  caffe_set(top[1]->count(), Dtype(-FLT_MAX), hash_table);
  //caffe_set(top[2]->count(), Dtype(-FLT_MAX), hash_bool_table);

  int num_small_roi = 0;
  int num_normal_roi = 0;
  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind = bottom_rois[0];
    CHECK_GE(roi_batch_ind, 0);
    //CHECK_LT(roi_batch_ind, batch_size);
    
    // padding
    Dtype pad_w, pad_h;
    pad_w = (bottom_rois[3]-bottom_rois[1]+1)*pad_ratio_;
    pad_h = (bottom_rois[4]-bottom_rois[2]+1)*pad_ratio_;
    int roi_start_w = round((bottom_rois[1]-pad_w) * spatial_scale_);
    int roi_start_h = round((bottom_rois[2]-pad_h) * spatial_scale_);
    int roi_end_w = round((bottom_rois[3]+pad_w) * spatial_scale_);
    int roi_end_h = round((bottom_rois[4]+pad_h) * spatial_scale_);
    // clipping
    roi_start_w = max(roi_start_w,0); roi_start_h = max(roi_start_h,0);
    int img_width = round(width_/spatial_scale_);
    int img_height = round(height_/spatial_scale_);
    roi_end_w = min(img_width-1,roi_end_w);
    roi_end_h = min(img_height-1,roi_end_h);
    
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    
    if (roi_height < pooled_height_ || roi_width < pooled_width_ )
    {
       //cout<<"num_small_roi"<<num_small_roi<<endl;
       hash_table[0] = static_cast<Dtype>(num_small_roi);
       hash_table[1] = static_cast<Dtype>(2);
       num_small_roi++;
    }
    else
    {
       //cout<<"num_normal_roi"<<num_normal_roi<<endl;
       hash_table[0] = static_cast<Dtype>(num_normal_roi);
       hash_table[1] = static_cast<Dtype>(1);
       num_normal_roi++;
    }

    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);  
    hash_table += top[1]->offset(1);
    //hash_bool_table += top[2]->offset(1);
  }
  //cout<<"num_small_roi: "<<num_small_roi<<endl;
  //cout<<"num_normal_roi: "<<num_normal_roi<<endl;

  top_data[0] = static_cast<Dtype>(num_small_roi);
  top_data += top[0]->offset(1);
  top_data[0] = static_cast<Dtype>(num_normal_roi);
}

template <typename Dtype>
void ROICountLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}


#ifdef CPU_ONLY
STUB_GPU(ROICountLayer);
#endif

INSTANTIATE_CLASS(ROICountLayer);
REGISTER_LAYER_CLASS(ROICount);

}  // namespace caffe
