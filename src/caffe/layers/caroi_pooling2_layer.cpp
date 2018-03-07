// Context-Aware ROI Pooling, 2016
// The Chinese University of Hong Kong
// Written by Hu Xiaowei
// -------------------------------------------------------------------------
// Thanks for Ross Girshick providing original ori_pooling code for reference [Fast R-CNN]
// [see fast-rcnn/LICENSE for details]
// -------------------------------------------------------------------------
#ifdef USE_OPENCV

#include "opencv2/imgproc/imgproc.hpp"

#include <cfloat>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <iostream>

using std::max;
using std::min;
using std::floor;
using std::ceil;
using std::cout;
using std::endl;

namespace caffe {

template <typename Dtype>
void CAROIPooling2Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //cout<<"Layer SetUp!!!!!!!!!!!!!!!!!!!!"<<endl;
  CAROIPooling2Parameter caroi_pool_param = this->layer_param_.caroi_pooling2_param();
  CHECK_GT(caroi_pool_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(caroi_pool_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = caroi_pool_param.pooled_h();
  pooled_width_ = caroi_pool_param.pooled_w();
  spatial_scale_ = caroi_pool_param.spatial_scale();
  pad_ratio_normal_ = caroi_pool_param.pad_ratio_normal();
  pad_ratio_small_ = caroi_pool_param.pad_ratio_small();
  normal_branch_ = caroi_pool_param.normal_branch();
  small_branch_ = caroi_pool_param.small_branch();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void CAROIPooling2Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //cout<<"Layer Reshape!!!!!!!!!!!!!!!!!!!!"<<endl;
  small_roi_ = static_cast<int>((bottom[2]->cpu_data())[0]);
  //cout<<small_roi_<<endl;

  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  
  int top0_roi_num = bottom[1]->num()-small_roi_;
  int top1_roi_num = small_roi_;
  if (small_roi_ == 0)
  {
     top1_roi_num = 1;
     small_roi_zero = true;
  }
  else
  {  small_roi_zero = false; }
  if (top0_roi_num == 0)
  {
     top0_roi_num = 1;
     normal_roi_zero = true;
  }
  else
  {  normal_roi_zero = false; }

  //cout<<"top0_roi_num: "<<top0_roi_num<<"  top1_roi_num: "<<top1_roi_num<<endl;
  if (normal_branch_)
  {
     top[0]->Reshape(top0_roi_num, channels_, pooled_height_,
        pooled_width_);
     if (small_branch_)
     {
        top[1]->Reshape(top1_roi_num, channels_, pooled_height_,
           pooled_width_);
     }
  }
  else if (small_branch_)
  {
      top[0]->Reshape(top1_roi_num, channels_, pooled_height_,
         pooled_width_);
  }
  else
  {   
      LOG(INFO) << "ERROR!";
  }
  max_idx_.Reshape(top0_roi_num + top1_roi_num, channels_, pooled_height_,
      pooled_width_);
}

template <typename Dtype>
void CAROIPooling2Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void CAROIPooling2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(CAROIPooling2Layer);
#endif

INSTANTIATE_CLASS(CAROIPooling2Layer);
REGISTER_LAYER_CLASS(CAROIPooling2);

}  // namespace caffe
#endif //USE_OPENCV
