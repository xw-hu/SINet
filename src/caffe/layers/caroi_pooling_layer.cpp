// Context-Aware ROI Pooling, 2016
// The Chinese University of Hong Kong
// Written by Hu Xiaowei
// -------------------------------------------------------------------------
// Thanks for Ross Girshick providing original ori_pooling code [Fast R-CNN]
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
void CAROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //cout<<"Layer SetUp!!!!!!!!!!!!!!!!!!!!"<<endl;
  CAROIPoolingParameter caroi_pool_param = this->layer_param_.caroi_pooling_param();
  CHECK_GT(caroi_pool_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(caroi_pool_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = caroi_pool_param.pooled_h();
  pooled_width_ = caroi_pool_param.pooled_w();
  spatial_scale_ = caroi_pool_param.spatial_scale();
  pad_ratio_ = caroi_pool_param.pad_ratio();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void CAROIPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //cout<<"Layer Reshape!!!!!!!!!!!!!!!!!!!!"<<endl;
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
  max_idx_.Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
}

template <typename Dtype>
void CAROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  //cout<<"Layer Forward!!!!!!!!!!!!!!!!!!!!"<<endl;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  // num of small ROIs (the size on feature maps is smaller than the bin)
  int num_small_rois = 0;

  // Number of ROIs
  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num();
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  int* argmax_data = max_idx_.mutable_cpu_data();
  caffe_set(top_count, -1, argmax_data);

  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind = bottom_rois[0];
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);
    
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

    //get the batch_data at coorect channel (first dim)
    const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind); //bottom[0]->cpu_data();
    
    const float bin_size_h_float = (float)roi_height / (float)pooled_height_;
    const float bin_size_w_float = (float)roi_width / (float)pooled_width_;
 
    //**********************CUHK HU XIAOWEI**************************************deal with small rois
    if (bin_size_h_float<1 || bin_size_w_float<1)
    {            
       //************************************************************2016.11 
       if (bin_size_h_float>1)  //height is large
       {
            num_small_rois++;
           int enlarge_pad_h = 0; //enlarge_pad = (multipler-1)
           int enlarge_pad_w = 0; //pooled_width_/roi_width;
           //cout<<"enlarge_pad: "<<enlarge_pad_h<<"  "<<enlarge_pad_w<<endl;
       
           cv::Mat ori_roi_feature(roi_height,roi_width,CV_32F); //add(0,2)
       
           cv::Size cv_enlarge_size;
           cv_enlarge_size.height = roi_height; //both sides (no padding)
           cv_enlarge_size.width = pooled_width_+enlarge_pad_w*2;

           cv::Mat enlarge_roi_feature(cv_enlarge_size,CV_32F); //+2 enlarged feature map

           for (int c = 0; c < channels_; ++c)  //the index to the correct channel
           {
               //int pad_h1=1,pad_h2=1,pad_w1=1,pad_w2=1;

               //cout<<"rows:"<<ori_roi_feature.rows<<" cols:"<<ori_roi_feature.cols<<" channels:"<<ori_roi_feature.channels()<<endl; // channel is 1
               for (int i = 0, ori_roi_h = roi_start_h; i < roi_height; i++, ori_roi_h++) //expand 1 pixel for both size 
               {
                   int h = min(max(ori_roi_h, 0), height_-1);  //check the border [nearest neighborhood if it's exceeded border]
                   for (int j = 0, ori_roi_w = roi_start_w; j < roi_width; j++, ori_roi_w++)
                   {
                       int w = min(max(ori_roi_w, 0), width_-1); //check the border [nearest neighborhood if it's exceeded border]
                       const int index = h * width_ + w;
                       ori_roi_feature.at<float>(i,j) = static_cast<float>(batch_data[index]);
                   }
               }
              
               //cout<<"M= "<<endl<< " "<<ori_roi_feature<<endl<<endl; 
               cv::resize(ori_roi_feature,enlarge_roi_feature,cv_enlarge_size,0,0,cv::INTER_LINEAR);
               //cout<<"XXM= "<<endl<< " "<<enlarge_roi_feature<<endl<<endl; 
               
               const Dtype bin_size_h = static_cast<Dtype>(bin_size_h_float);
               const Dtype bin_size_w = static_cast<Dtype>(1.0); //bin size for enlarged width is 1
                
               for (int ph = 0; ph < pooled_height_; ++ph) {
                    for (int pw = 0; pw < pooled_width_; ++pw) {
                         int hstart = static_cast<int>(floor(static_cast<Dtype>(ph) * bin_size_h));
                         int wstart = static_cast<int>(floor(static_cast<Dtype>(pw) * bin_size_w));
                         int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1) * bin_size_h));
                         int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1) * bin_size_w));

                         hstart = min(max(hstart + enlarge_pad_h, 0), height_);
                         hend = min(max(hend + enlarge_pad_h, 0), height_);
                         wstart = min(max(wstart + enlarge_pad_w, 0), width_);
                         wend = min(max(wend + enlarge_pad_w, 0), width_);
                        

                         bool is_empty = (hend <= hstart) || (wend <= wstart);

                         const int pool_index = ph * pooled_width_ + pw;
                         if (is_empty) {
                               top_data[pool_index] = 0;
                               argmax_data[pool_index] = -1;
                         }

                         // cout<<"hstart:"<<hstart<<endl<<"hend:"<<hend<<endl;
                         for (int h = hstart; h < hend; ++h) {
                             for (int w = wstart; w < wend; ++w) {
                                  const int index = h * cv_enlarge_size.width  + w; 
                                  if (static_cast<Dtype>(enlarge_roi_feature.at<float>(h,w)) > top_data[pool_index]) {
                                      top_data[pool_index] = static_cast<Dtype>(enlarge_roi_feature.at<float>(h,w));
                                      argmax_data[pool_index] = index; //index on the enlarged feature map  //the index is based on new enlarged feature map
                                  }
                             }
                         }
                    }
               }
               // Increment all data pointers by one channel
               batch_data += bottom[0]->offset(0, 1);
               top_data += top[0]->offset(0, 1);
               argmax_data += max_idx_.offset(0, 1);
            }
       }
       else if (bin_size_w_float>1)  //weight is large
       {
           num_small_rois++;
           int enlarge_pad_h = 0;//pooled_height_/roi_height; //enlarge_pad = (multipler-1)
           int enlarge_pad_w = 0;
           //cout<<"enlarge_pad: "<<enlarge_pad_h<<"  "<<enlarge_pad_w<<endl;
       
           cv::Mat ori_roi_feature(roi_height,roi_width,CV_32F); //add(2,0)
       
           cv::Size cv_enlarge_size;
           cv_enlarge_size.height = pooled_height_+enlarge_pad_h*2; //both sides (no padding)
           cv_enlarge_size.width = roi_width;

           cv::Mat enlarge_roi_feature(cv_enlarge_size,CV_32F); //+2 enlarged feature map

           for (int c = 0; c < channels_; ++c)  //the index to the correct channel
           {
               //int pad_h1=1,pad_h2=1,pad_w1=1,pad_w2=1;

               //cout<<"rows:"<<ori_roi_feature.rows<<" cols:"<<ori_roi_feature.cols<<" channels:"<<ori_roi_feature.channels()<<endl; // channel is 1
               for (int i = 0, ori_roi_h = roi_start_h; i < roi_height; i++, ori_roi_h++) //expand 1 pixel for both size 
               {
                   int h = min(max(ori_roi_h, 0), height_-1);  //check the border [nearest neighborhood if it's exceeded border]
                   for (int j = 0, ori_roi_w = roi_start_w; j < roi_width; j++, ori_roi_w++)
                   {
                       int w = min(max(ori_roi_w, 0), width_-1); //check the border [nearest neighborhood if it's exceeded border]
                       const int index = h * width_ + w;
                       ori_roi_feature.at<float>(i,j) = static_cast<float>(batch_data[index]);
                   }
               }
               
               //cout<<"M= "<<endl<< " "<<ori_roi_feature<<endl<<endl; 
               cv::resize(ori_roi_feature,enlarge_roi_feature,cv_enlarge_size,0,0,cv::INTER_LINEAR);
               //cout<<"XXM= "<<endl<< " "<<enlarge_roi_feature<<endl<<endl; 
               
               const Dtype bin_size_h = static_cast<Dtype>(1.0);
               const Dtype bin_size_w = static_cast<Dtype>(bin_size_w_float); //bin size for enlarged width is 1
                
               for (int ph = 0; ph < pooled_height_; ++ph) {
                    for (int pw = 0; pw < pooled_width_; ++pw) {
                         int hstart = static_cast<int>(floor(static_cast<Dtype>(ph) * bin_size_h));
                         int wstart = static_cast<int>(floor(static_cast<Dtype>(pw) * bin_size_w));
                         int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1) * bin_size_h));
                         int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1) * bin_size_w));

                         hstart = min(max(hstart + enlarge_pad_h, 0), height_);
                         hend = min(max(hend + enlarge_pad_h, 0), height_);
                         wstart = min(max(wstart + enlarge_pad_w, 0), width_);
                         wend = min(max(wend + enlarge_pad_w, 0), width_);
                         //cout<<"hstart: "<<hstart<<" hend: "<<hend<<" wstart: "<<wstart<<" wend: "<<wend<<endl;

                         bool is_empty = (hend <= hstart) || (wend <= wstart);

                         const int pool_index = ph * pooled_width_ + pw;
                         if (is_empty) {
                               top_data[pool_index] = 0;
                               argmax_data[pool_index] = -1;
                         }

                         // cout<<"hstart:"<<hstart<<endl<<"hend:"<<hend<<endl;
                         for (int h = hstart; h < hend; ++h) {
                             for (int w = wstart; w < wend; ++w) {
                                  const int index = h * cv_enlarge_size.width  + w; 
                                  if (static_cast<Dtype>(enlarge_roi_feature.at<float>(h,w)) > top_data[pool_index]) {
                                      top_data[pool_index] = static_cast<Dtype>(enlarge_roi_feature.at<float>(h,w));
                                      argmax_data[pool_index] = index; //index on the enlarged feature map  //the index is based on new enlarged feature map
                                  }
                             }
                         }
                    }
               }
               // Increment all data pointers by one channel
               batch_data += bottom[0]->offset(0, 1);
               top_data += top[0]->offset(0, 1);
               argmax_data += max_idx_.offset(0, 1);
            }
       }
       //************************************************************2016.11
       else
       {
           num_small_rois++;
           int enlarge_pad_h = 0;//pooled_height_/roi_height; //enlarge_pad = (multipler-1)
           int enlarge_pad_w = 0;//pooled_width_/roi_width;
           //cout<<"enlarge_pad: "<<enlarge_pad_h<<"  "<<enlarge_pad_w<<endl;
       
           cv::Mat ori_roi_feature(roi_height,roi_width,CV_32F); //
       
           cv::Size cv_enlarge_size;
           cv_enlarge_size.height = pooled_height_+enlarge_pad_h*2; //both sides 
           cv_enlarge_size.width = pooled_width_+enlarge_pad_w*2;

           cv::Mat enlarge_roi_feature(cv_enlarge_size,CV_32F); //+2 enlarged feature map

           for (int c = 0; c < channels_; ++c)  //the index to the correct channel
           {
               //int pad_h1=1,pad_h2=1,pad_w1=1,pad_w2=1;

               //cout<<"rows:"<<ori_roi_feature.rows<<" cols:"<<ori_roi_feature.cols<<" channels:"<<ori_roi_feature.channels()<<endl; // channel is 1
               for (int i = 0, ori_roi_h = roi_start_h; i < roi_height; i++, ori_roi_h++) //expand 1 pixel for both size 
               {
                   int h = min(max(ori_roi_h, 0), height_-1);  //check the border [nearest neighborhood if it's exceeded border]
                   for (int j = 0, ori_roi_w = roi_start_w; j < roi_width; j++, ori_roi_w++)
                   {
                       int w = min(max(ori_roi_w, 0), width_-1); //check the border [nearest neighborhood if it's exceeded border]
                       const int index = h * width_ + w;
                       ori_roi_feature.at<float>(i,j) = static_cast<float>(batch_data[index]);
                   }
               }
             
               //cout<<"M= "<<endl<< " "<<ori_roi_feature<<endl<<endl; 
               cv::resize(ori_roi_feature,enlarge_roi_feature,cv_enlarge_size,0,0,cv::INTER_LINEAR);
               //cout<<"XXM= "<<endl<< " "<<enlarge_roi_feature<<endl<<endl; 

               for (int ph = 0; ph < pooled_height_; ++ph) { //skip the pad values
                   int matrix_h = ph + enlarge_pad_h; //index for the enlarged matrix
               
                   for (int pw = 0; pw < pooled_width_; ++pw) {
                       int matrix_w = pw + enlarge_pad_w;
                       const int pool_index = ph * pooled_width_ + pw; 
                    
                       //cout<<enlarge_roi_feature.at<float>(matrix_h,matrix_w)<<endl;
                       top_data[pool_index] = static_cast<Dtype>(enlarge_roi_feature.at<float>(matrix_h,matrix_w));//batch_data[index];

                       //cout<<top_data[pool_index]<<endl;
                   }
               }
               // Increment all data pointers by one channel
               batch_data += bottom[0]->offset(0, 1);
               top_data += top[0]->offset(0, 1);
               argmax_data += max_idx_.offset(0, 1);
           }
       }
       // Increment ROI data pointer
       bottom_rois += bottom[1]->offset(1);
       continue;
    }
    //**********************CUHK HU XIAOWEI**************************************deal with small rois
    
    const Dtype bin_size_h = static_cast<Dtype>(bin_size_h_float);
    const Dtype bin_size_w = static_cast<Dtype>(bin_size_w_float);
    
    for (int c = 0; c < channels_; ++c) { //the index to the correct channel
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                              * bin_size_h));
          int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                              * bin_size_w));
          int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                           * bin_size_h));
          int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                           * bin_size_w));

          hstart = min(max(hstart + roi_start_h, 0), height_);
          hend = min(max(hend + roi_start_h, 0), height_);
          wstart = min(max(wstart + roi_start_w, 0), width_);
          wend = min(max(wend + roi_start_w, 0), width_);

          bool is_empty = (hend <= hstart) || (wend <= wstart);

          const int pool_index = ph * pooled_width_ + pw;
          if (is_empty) {
            top_data[pool_index] = 0;
            argmax_data[pool_index] = -1;
          }

         // cout<<"hstart:"<<hstart<<endl<<"hend:"<<hend<<endl;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = h * width_ + w;
              if (batch_data[index] > top_data[pool_index]) {
                top_data[pool_index] = batch_data[index];
                argmax_data[pool_index] = index;
              }
            }
          }
        }
      }
      // Increment all data pointers by one channel
      batch_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
      argmax_data += max_idx_.offset(0, 1);
    }
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }
}

template <typename Dtype>
void CAROIPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(CAROIPoolingLayer);
#endif

INSTANTIATE_CLASS(CAROIPoolingLayer);
REGISTER_LAYER_CLASS(CAROIPooling);

}  // namespace caffe
#endif //USE_OPENCV
