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

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <iostream>

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void CAROIPoolForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const Dtype pad_ratio, const Dtype* bottom_rois, Dtype* top_data, int* argmax_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    //printf("Forward!!!!\n");
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];

    // padding
    Dtype pad_w, pad_h;
    pad_w = (bottom_rois[3]-bottom_rois[1]+1)*pad_ratio;
    pad_h = (bottom_rois[4]-bottom_rois[2]+1)*pad_ratio;
    int roi_start_w = round((bottom_rois[1]-pad_w) * spatial_scale);
    int roi_start_h = round((bottom_rois[2]-pad_h) * spatial_scale);
    int roi_end_w = round((bottom_rois[3]+pad_w) * spatial_scale);
    int roi_end_h = round((bottom_rois[4]+pad_h) * spatial_scale);
    // clipping
    roi_start_w = max(roi_start_w,0); roi_start_h = max(roi_start_h,0);
    int img_width = round(width / spatial_scale);
    int img_height = round(height / spatial_scale);
    roi_end_w = min(img_width-1,roi_end_w);
    roi_end_h = min(img_height-1,roi_end_h);

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    
    float bin_size_h_float = (float)roi_height / (float)pooled_height;
    float bin_size_w_float = (float)roi_width / (float)pooled_width;

    //**********************CUHK HU XIAOWEI**************************************deal with small rois
    if (bin_size_h_float<1 || bin_size_w_float<1)
    {
       if (bin_size_h_float>1)  //height is large
       {
           int enlarge_pad_h = 0; //enlarge_pad = (multipler-1)
           int enlarge_pad_w = 0;//pooled_width/roi_width;
           int enlarge_height = roi_height; 
           int enlarge_width = pooled_width+enlarge_pad_w*2; //both sides 

           int ori_roi_h,ori_roi_w,h,w;
           int M1=roi_width, M2=roi_height;  //old matrix width/height (non-enlarge for height)
           
           //bilinear interpolation (just on the x direction)
           //value = bilinear_interpolation_kernel_GPU(ori_roi_feature, roi_width+2, roi_height+2, enlarge_width, enlarge_height, matrix_w, matrix_h);      
           float x_ratio = ((float)(M1-1))/(enlarge_width-1);
           float y_ratio = ((float)(M2-1))/(enlarge_height-1);  
         
           float d00, d10;  
           
           //find the largest value in the bin
           Dtype bin_size_h = static_cast<Dtype>(bin_size_h_float);
           Dtype bin_size_w = static_cast<Dtype>(1.0); //1.0

           int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                         * bin_size_h));
           int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                         * bin_size_w));
           int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                         * bin_size_h));
           int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                         * bin_size_w));
       
           // Add roi offsets and clip to input boundaries       
           hstart = min(max(hstart, 0), height);
           hend = min(max(hend, 0), height);
           wstart = min(max(wstart, 0), width);  
           wend = min(max(wend, 0), width);    
           bool is_empty = (hend <= hstart) || (wend <= wstart);

           // Define an empty pooling region to be zero
           Dtype maxval = is_empty ? 0 : -FLT_MAX;
           // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
           int maxidx = -1;
           bottom_data += (roi_batch_ind * channels + c) * height * width;  //get the bottom data (feature map)
           
           for (int hi = hstart; hi < hend; ++hi) {
               for (int wi = wstart; wi < wend; ++wi) {
                   int bottom_index = hi * width + wi; //on the original feature map [bin_size_w=1.0]
                   
                   ////////////////////////////////////////////////////////////////do interpolation for every data point [just along the x direction]
                   int matrix_h = hi + enlarge_pad_h; //index for the enlarged matrix [only this point need to be interpolated]
                   int matrix_w = wi + enlarge_pad_w;
                   const int    ind_x = floor(x_ratio * matrix_w); 
                   const float  a     = x_ratio * matrix_w-ind_x; //0-1 ratio in []

                   const int    ind_y = floor(y_ratio * matrix_h); 

                   if (((ind_x)   < M1)&&((ind_y)   < M2))  {      
                      ori_roi_h = roi_start_h+ind_y;    //no pad
                      ori_roi_w = roi_start_w+ind_x;  //pad 0
                      h = min(max(ori_roi_h, 0), height-1); 
                      w = min(max(ori_roi_w, 0), width-1);
                      d00=bottom_data[h * width + w]; //get the feature value
                   }  else{ d00 = 0.0; }

                   if (((ind_x+1) < M1)&&((ind_y)   < M2))  {      
                      ori_roi_h = roi_start_h+ind_y;     //no pad
                      ori_roi_w = roi_start_w+ind_x+1; //pad 0
                      h = min(max(ori_roi_h, 0), height-1); 
                      w = min(max(ori_roi_w, 0), width-1);
                      d10=bottom_data[h * width + w]; //get the feature value
                    }  else{ d10 = 0.0; }

                    Dtype vaule = static_cast<Dtype>(a * d10 + (-d00 * a + d00));  //a=(x-x1)/(x2-x1)               
                   ////////////////////////////////////////////////////////////////

                   if (vaule > maxval) {
                      maxval = vaule; //bottom_data[bottom_index];
                      maxidx = bottom_index;
                   }
               }
           }
           top_data[index] = maxval;
           argmax_data[index] = maxidx;
       }
       else if (bin_size_w_float>1)  //weight is large
       {
           int enlarge_pad_h = 0;//pooled_height/roi_height; //enlarge_pad = (multipler-1)
           int enlarge_pad_w = 0;
           int enlarge_height = pooled_height+enlarge_pad_h*2; //both sides 
           int enlarge_width = roi_width;

           int ori_roi_h,ori_roi_w,h,w;
           int M1=roi_width, M2=roi_height;  //old matrix width/height (non-enlarge for height)
           
           //bilinear interpolation (just on the x direction)
           //value = bilinear_interpolation_kernel_GPU(ori_roi_feature, roi_width+2, roi_height+2, enlarge_width, enlarge_height, matrix_w, matrix_h);      
           float x_ratio = ((float)(M1-1))/(enlarge_width-1);
           float y_ratio = ((float)(M2-1))/(enlarge_height-1);  
         
           float d00, d01;  
           
           //find the largest value in the bin
           Dtype bin_size_h = static_cast<Dtype>(1.0);
           Dtype bin_size_w = static_cast<Dtype>(bin_size_w_float);

           int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                         * bin_size_h));
           int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                         * bin_size_w));
           int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                         * bin_size_h));
           int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                         * bin_size_w));
       
           // Add roi offsets and clip to input boundaries       
           hstart = min(max(hstart, 0), height);
           hend = min(max(hend, 0), height);
           wstart = min(max(wstart, 0), width);  
           wend = min(max(wend, 0), width);    
           bool is_empty = (hend <= hstart) || (wend <= wstart);

           // Define an empty pooling region to be zero
           Dtype maxval = is_empty ? 0 : -FLT_MAX;
           // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
           int maxidx = -1;
           bottom_data += (roi_batch_ind * channels + c) * height * width;  //get the bottom data (feature map)
          
           for (int hi = hstart; hi < hend; ++hi) {
               for (int wi = wstart; wi < wend; ++wi) {
                   int bottom_index = hi * width + wi; //on the original feature map

                   ////////////////////////////////////////////////////////////////do interpolation for every data point [just along the x direction]
                   int matrix_h = hi + enlarge_pad_h; //index for the enlarged matrix [only this point need to be interpolated]
                   int matrix_w = wi + enlarge_pad_w;
                   const int    ind_x = floor(x_ratio * matrix_w); 

                   const int    ind_y = floor(y_ratio * matrix_h); 
                   const float  b     = y_ratio * matrix_h-ind_y;

                   if (((ind_x)   < M1)&&((ind_y)   < M2))  {      
                      ori_roi_h = roi_start_h+ind_y;    //pad 0
                      ori_roi_w = roi_start_w+ind_x;  //no pad 
                      h = min(max(ori_roi_h, 0), height-1); 
                      w = min(max(ori_roi_w, 0), width-1);
                      d00=bottom_data[h * width + w]; //get the feature value
                   }  else{ d00 = 0.0; }

                  if (((ind_x)   < M1)&&((ind_y+1) < M2))  {      
                      ori_roi_h = roi_start_h+ind_y+1;
                      ori_roi_w = roi_start_w+ind_x;  //no pad
                      h = min(max(ori_roi_h, 0), height-1); 
                      w = min(max(ori_roi_w, 0), width-1);
                      d01=bottom_data[h * width + w]; //get the feature value
                   }  else{ d01 = 0.0; }
                    
                    Dtype vaule = static_cast<Dtype>(b * d01 + (-d00 * b + d00));
                   ////////////////////////////////////////////////////////////////

                   if (vaule > maxval) {
                      maxval = vaule; //bottom_data[bottom_index];
                      maxidx = bottom_index;
                   }
               }
           }
           top_data[index] = maxval;
           argmax_data[index] = maxidx;
       }
       else
       {
           //num_small_rois++;
           int enlarge_pad_h = 0;//pooled_height/roi_height; //enlarge_pad = (multipler-1)
           int enlarge_pad_w = 0;//pooled_width/roi_width;
           int enlarge_height = pooled_height+enlarge_pad_h*2; //both sides 
           int enlarge_width = pooled_width+enlarge_pad_w*2;
       
           bottom_data += (roi_batch_ind * channels + c) * height * width ;//get the bottom data (feature map)
        
           //float ori_roi_feature[400];
           //roi feature map initialization
           //float *ori_roi_feature;
           //ori_roi_feature = new float [(roi_height+2)*(roi_width+2)];
      
           /*for (int i = 0, ori_roi_h = roi_start_h-1; i < roi_height+2; i++, ori_roi_h++) //expand 1 pixel for both size 
           {
               int h = min(max(ori_roi_h, 0), height-1);  //check the border [nearest neighborhood if it's exceeded border]
               for (int j = 0, ori_roi_w = roi_start_w-1; j < roi_width+2; j++, ori_roi_w++)
               {
                   int w = min(max(ori_roi_w, 0), width-1); //check the border [nearest neighborhood if it's exceeded border]
                   int bottom_index = h * width + w;
                   ori_roi_feature[i*(roi_width+2)+j] = bottom_data[bottom_index];
               }
           }*/

           int matrix_h = ph + enlarge_pad_h; //index for the enlarged matrix [only this point need to be interpolated]
           int matrix_w = pw + enlarge_pad_w;
           int ori_roi_h,ori_roi_w,h,w;
           int M1=roi_width, M2=roi_height;  //old matrix width/height

           //bilinear interpolation
           //value = bilinear_interpolation_kernel_GPU(ori_roi_feature, roi_width+2, roi_height+2, enlarge_width, enlarge_height, matrix_w, matrix_h);      
           float x_ratio = ((float)(M1-1))/(enlarge_width-1);
           float y_ratio = ((float)(M2-1))/(enlarge_height-1);  
   
           float result_temp1, result_temp2;

           const int    ind_x = floor(x_ratio * matrix_w); 
           const float  a     = x_ratio * matrix_w-ind_x; //0-1 ratio in []

           const int    ind_y = floor(y_ratio * matrix_h); 
           const float  b     = y_ratio * matrix_h-ind_y;

           float d00, d01, d10, d11;
       
           if (((ind_x)   < M1)&&((ind_y)   < M2))  {      
              ori_roi_h = roi_start_h+ind_y;
              ori_roi_w = roi_start_w+ind_x;
              h = min(max(ori_roi_h, 0), height-1); 
              w = min(max(ori_roi_w, 0), width-1);
              d00=bottom_data[h * width + w]; //get the feature value
           }  else{ d00 = 0.0; }

           if (((ind_x+1) < M1)&&((ind_y)   < M2))  {      
              ori_roi_h = roi_start_h+ind_y;
              ori_roi_w = roi_start_w+ind_x+1;
              h = min(max(ori_roi_h, 0), height-1); 
              w = min(max(ori_roi_w, 0), width-1);
              d10=bottom_data[h * width + w]; //get the feature value
           }  else{ d10 = 0.0; }
       
           if (((ind_x)   < M1)&&((ind_y+1) < M2))  {      
              ori_roi_h = roi_start_h+ind_y+1;
              ori_roi_w = roi_start_w+ind_x;
              h = min(max(ori_roi_h, 0), height-1); 
              w = min(max(ori_roi_w, 0), width-1);
              d01=bottom_data[h * width + w]; //get the feature value
           }  else{ d01 = 0.0; }
       
           if (((ind_x+1) < M1)&&((ind_y+1) < M2))  {      
              ori_roi_h = roi_start_h+ind_y+1;
              ori_roi_w = roi_start_w+ind_x+1;
              h = min(max(ori_roi_h, 0), height-1); 
              w = min(max(ori_roi_w, 0), width-1);
              d11=bottom_data[h * width + w]; //get the feature value
           }  else{ d11 = 0.0; }

           result_temp1 = a * d10 + (-d00 * a + d00); //a=(x-x1)/(x2-x1)

           result_temp2 = a * d11 + (-d01 * a + d01);

           top_data[index] = static_cast<Dtype>(b * result_temp2 + (-result_temp1 * b + result_temp1)); //final interpolation result
           argmax_data[index] = -2; //-2 index interpolation
       
       }
       return; 
    }
     
    //**********************CUHK HU XIAOWEI**************************************deal with small rois
    
    Dtype bin_size_h = static_cast<Dtype>(bin_size_h_float);
    Dtype bin_size_w = static_cast<Dtype>(bin_size_w_float);

    int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                        * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                        * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h * width + w;
        if (bottom_data[bottom_index] > maxval) {
          maxval = bottom_data[bottom_index];
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    argmax_data[index] = maxidx;
  }
}

template <typename Dtype>
void CAROIPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int* argmax_data = max_idx_.mutable_gpu_data();
  int count = top[0]->count();
  
  // num of small ROIs (the size on feature maps is smaller than the bin)
  //int& num_small_rois = 0;
  // NOLINT_NEXT_LINE(whitespace/operators)
  CAROIPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, spatial_scale_, channels_, height_, width_,
      pooled_height_, pooled_width_, pad_ratio_, bottom_rois, top_data, argmax_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void CAROIPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int num_rois, const Dtype spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const Dtype pad_ratio, 
    Dtype* bottom_diff, const Dtype* bottom_rois) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    //printf("Backward!!!!");
    // (n, c, h, w) coords in bottom data
    int w = index % width;  //original (unpooled index)
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    
    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      // padding
      Dtype pad_w, pad_h;
      pad_w = (offset_bottom_rois[3]-offset_bottom_rois[1]+1)*pad_ratio;
      pad_h = (offset_bottom_rois[4]-offset_bottom_rois[2]+1)*pad_ratio;
      int roi_start_w = round((offset_bottom_rois[1]-pad_w) * spatial_scale);
      int roi_start_h = round((offset_bottom_rois[2]-pad_h) * spatial_scale);
      int roi_end_w = round((offset_bottom_rois[3]+pad_w) * spatial_scale);
      int roi_end_h = round((offset_bottom_rois[4]+pad_h) * spatial_scale);
      // clipping
      roi_start_w = max(roi_start_w,0); roi_start_h = max(roi_start_h,0);
      int img_width = round(width / spatial_scale);
      int img_height = round(height / spatial_scale);
      roi_end_w = min(img_width-1,roi_end_w);
      roi_end_h = min(img_height-1,roi_end_h);


      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                           h >= roi_start_h && h <= roi_end_h);
      if (!in_roi) {
        continue;
      }
     
      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const Dtype* offset_top_diff = top_diff + offset;
      const int* offset_argmax_data = argmax_data + offset;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      // Force malformed ROIs to be 1x1
      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      int roi_height = max(roi_end_h - roi_start_h + 1, 1);
      
      float bin_size_h_float = (float)roi_height / (float)pooled_height;
      float bin_size_w_float = (float)roi_width / (float)pooled_width;
       
      //**********************CUHK HU XIAOWEI**************************************deal with small rois
      if (bin_size_h_float<1 || bin_size_w_float<1)
      {
         if (bin_size_h_float>1)  //height is large
         { 
             Dtype bin_size_h = static_cast<Dtype>(bin_size_h_float);
             Dtype bin_size_w = static_cast<Dtype>(1.0);

             int phstart = floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
             int phend = ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
             int pwstart = floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
             int pwend = ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);

             phstart = min(max(phstart, 0), pooled_height);
             phend = min(max(phend, 0), pooled_height);
             pwstart = min(max(pwstart, 0), pooled_width);
             pwend = min(max(pwend, 0), pooled_width);

             for (int ph = phstart; ph < phend; ++ph) {
               for (int pw = pwstart; pw < pwend; ++pw) {
                  if (offset_argmax_data[ph * pooled_width + pw] == (h * width + pw)) {  // pw -> bin_size_w=1.0
                     
                     /////////////////////////////////////////////////////////////////////////////the max value (do interpolation)
                     int w_index = w - roi_start_w;  //position need to be assigned a difference
                     
                     //weighted sum diff
	             float x_ratio = ((float)(pooled_width-1))/(float)(roi_width-1);
	             int pool_pad_w = (float)pooled_width/(float)roi_width;

                     float ind_x = x_ratio * w_index; //in the pooled coordinate
	             float diff=0;
                     
                     for (int j=0; j<=((floor(ind_x)==ind_x)?pool_pad_w:pool_pad_w-1); j++)
	             {
	                 float weight_w1,weight_w2;
		         int w1=floor(ind_x)-j, w2=ceil(ind_x)+j; //pooled coordinate

		         if (w1!=w2) //w1 & w2 not at the same position
	 	         {  
		             weight_w1=(x_ratio-(ind_x-w1))/x_ratio, weight_w2=(x_ratio-(w2-ind_x))/x_ratio; //x_ratio is the distance to the next point
		         }
		         else
		         {
		             weight_w1=0.5, weight_w2=0.5; //at the same row
		         }

		         float diff1,diff2;  //points which are symmetric around center point [x axis]
		         diff1 = (w1<0)? 0 : offset_top_diff[ph*pooled_width+w1];
		         diff2 = (w2>=pooled_width)? 0 : offset_top_diff[ph*pooled_width+w2];
		         diff += weight_w1*diff1 + weight_w2*diff2;
	                
	             }
	             gradient += diff;              
                     /////////////////////////////////////////////////////////////////////////////////////////////
                  }
               }
             }
         }
         else if (bin_size_w_float>1)
         {
             Dtype bin_size_h = static_cast<Dtype>(1.0);
             Dtype bin_size_w = static_cast<Dtype>(bin_size_w_float);

             int phstart = floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
             int phend = ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
             int pwstart = floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
             int pwend = ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);

             phstart = min(max(phstart, 0), pooled_height);
             phend = min(max(phend, 0), pooled_height);
             pwstart = min(max(pwstart, 0), pooled_width);
             pwend = min(max(pwend, 0), pooled_width);

             for (int ph = phstart; ph < phend; ++ph) {
               for (int pw = pwstart; pw < pwend; ++pw) {
                  if (offset_argmax_data[ph * pooled_width + pw] == (ph * width + w)) {   // ph -> bin_size_h=1.0
                  
                     ////////////////////////////////////////////////////////////////////////////the max value (do interpolation) 
                     int h_index=h-roi_start_h; //position need to be assigned a difference
          
                     //weighted sum diff
                     float y_ratio = ((float)(pooled_height-1))/(float)(roi_height-1);  
	             int pool_pad_h = (float)pooled_height/(float)roi_height; //in this square, diff give weigths to the center point (pool_pad_h extention)

                     float ind_y = y_ratio * h_index; //in the pooled coordinate
           	     float diff=0;
  
                     for (int i=0; i<=((floor(ind_y)==ind_y)?pool_pad_h:pool_pad_h-1); i++) //deal with this ponit (surrounding points which give contribution to it)
                     {
                         float weight_h1, weight_h2;
	                 int h1=floor(ind_y)-i, h2=ceil(ind_y)+i; //pooled coordinate
	                 if (h1!= h2) //h1 & h2 are not at the same position
	                 {
                             weight_h1=(y_ratio-(ind_y-h1))/y_ratio, weight_h2=(y_ratio-(h2-ind_y))/y_ratio; //y_ratio is the distance to the next point
                         }
	                 else
	                 {
		             weight_h1=0.5, weight_h2=0.5; //at the same column
	                 }
	        
		         float diff1,diff3;  //four points which are symmetric around center point
		         diff1 = (h1<0)? 0 : offset_top_diff[h1*pooled_width+pw];
		         diff3 = (h2>=pooled_height)? 0 : offset_top_diff[h2*pooled_width+pw];
 		         diff += weight_h1*diff1 + weight_h2*diff3;
	             }
	             gradient += diff;          
                     /////////////////////////////////////////////////////////////////////////////////////////////
                  }
               }
             }
         }
         else
         {
         //float *pooled_roi_diff;
         //pooled_roi_diff = new float [pooled_height*pooled_width];
         
         //for (int i=0; i<pooled_height; i++){
         //    for (int j=0; j<pooled_width; j++){
         //        pooled_roi_diff[i * pooled_width + j]=offset_top_diff[i * pooled_width + j];
         //    }
         //}
         
            int h_index=h-roi_start_h; //position need to be assigned a difference
            int w_index=w-roi_start_w;
          
            //weighted sum diff
            float y_ratio = ((float)(pooled_height-1))/(float)(roi_height-1);  
	    float x_ratio = ((float)(pooled_width-1))/(float)(roi_width-1);

	    int pool_pad_h = (float)pooled_height/(float)roi_height; //in this square, diff give weigths to the center point (pool_pad_h extention)
	    int pool_pad_w = (float)pooled_width/(float)roi_width;

            float ind_y = y_ratio * h_index; //in the pooled coordinate
            float ind_x = x_ratio * w_index; //in the pooled coordinate
	    float diff=0;
  
            for (int i=0; i<=((floor(ind_y)==ind_y)?pool_pad_h:pool_pad_h-1); i++) //deal with this ponit (surrounding points which give contribution to it)
            {
                float weight_h1, weight_h2;
	        int h1=floor(ind_y)-i, h2=ceil(ind_y)+i; //pooled coordinate
	        if (h1!= h2) //h1 & h2 are not at the same position
	        {
                   weight_h1=(y_ratio-(ind_y-h1))/y_ratio, weight_h2=(y_ratio-(h2-ind_y))/y_ratio; //y_ratio is the distance to the next point
                }
	        else
	        {
		   weight_h1=0.5, weight_h2=0.5; //at the same column
	        }
	        for (int j=0; j<=((floor(ind_x)==ind_x)?pool_pad_w:pool_pad_w-1); j++)
	        {
		    float weight_w1,weight_w2;
		    int w1=floor(ind_x)-j, w2=ceil(ind_x)+j; //pooled coordinate
		    if (w1!=w2) //w1 & w2 not at the same position
	 	    { 
		       weight_w1=(x_ratio-(ind_x-w1))/x_ratio, weight_w2=(x_ratio-(w2-ind_x))/x_ratio; //x_ratio is the distance to the next point
		    }
		    else
		    {
		       weight_w1=0.5, weight_w2=0.5; //at the same row
		    }

		    float diff1,diff2,diff3,diff4;  //four points which are symmetric around center point
		    diff1 = (h1<0||w1<0)? 0 : offset_top_diff[h1*pooled_width+w1];
		    diff2 = (h1<0||w2>=pooled_width)? 0 : offset_top_diff[h1*pooled_width+w2];
		    diff3 = (h2>=pooled_height||w1<0)? 0 : offset_top_diff[h2*pooled_width+w1];
		    diff4 = (h2>=pooled_height||w2>=pooled_width)? 0 : offset_top_diff[h2*pooled_width+w2];

		    diff += weight_h1*weight_w1*diff1 + weight_h1*weight_w2*diff2 + weight_h2*weight_w1*diff3 + weight_h2*weight_w2*diff4;
	         }
	     }
	     gradient += diff;

             /*//bilinear interpolation (average pooling)
             float ratio_h = (float)pooled_height/(float)roi_height;
             float ratio_w = (float)pooled_width/(float)roi_width;
             int pool_stride_h = ratio_h; //the stride (height) every step moved  
             int pool_stride_w = ratio_w;
             int pool_h = ceil(ratio_h); //the height of bin in pooled map
             int pool_w = ceil(ratio_w);

             int poo_height_start = h_index*pool_stride_h; //the position for the bottom feature map need to be calculated
             int poo_width_start = w_index*pool_stride_w;
             Dtype value=0;
             for (int bin_i=poo_height_start; bin_i<poo_height_start+pool_h; bin_i++)
             {
                 for (int bin_j=poo_width_start; bin_j<poo_width_start+pool_w; bin_j++)
                 {
                      value += offset_top_diff[bin_i*pooled_width+bin_j];
                 }
             } 
             gradient += value/(static_cast<Dtype>(pool_h*pool_w)); //get the average value
         
             //bottom_diff[index] = gradient; //get the difference for this position
         
             //delete []pooled_roi_diff;
             */
         }
         continue;
      }
      //**********************CUHK HU XIAOWEI**************************************deal with small rois

      Dtype bin_size_h = static_cast<Dtype>(bin_size_h_float);
      Dtype bin_size_w = static_cast<Dtype>(bin_size_w_float);

      int phstart = floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
      int phend = ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
      int pwstart = floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
      int pwend = ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);

      phstart = min(max(phstart, 0), pooled_height);
      phend = min(max(phend, 0), pooled_height);
      pwstart = min(max(pwstart, 0), pooled_width);
      pwend = min(max(pwend, 0), pooled_width);

      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (offset_argmax_data[ph * pooled_width + pw] == (h * width + w)) {
            gradient += offset_top_diff[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void CAROIPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  const int* argmax_data = max_idx_.gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  CAROIPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, argmax_data, top[0]->num(), spatial_scale_, channels_,
      height_, width_, pooled_height_, pooled_width_, pad_ratio_, bottom_diff, bottom_rois);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(CAROIPoolingLayer);

}  // namespace caffe
#endif //USE_OPENCV
