// ------------------------------------------------------------------
// Copyright (c) 2016 
// The Chinese University of Hong Kong
// Written by Hu Xiaowei
// ------------------------------------------------------------------

#include <cfloat>
#include <ctime>
#include <cmath>

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
void ROISplitLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ROISplitParameter roi_split_param = this->layer_param_.roi_split_param();

  branch_num_ = roi_split_param.branch_num();

  CHECK_GT(branch_num_, 1)
      << "split_area must be > 1";
  CHECK_LT(branch_num_, 4)
      << "split_area must be < 4";

  split_area1_ = roi_split_param.split_area1();

  if (branch_num_ == 3)
  { split_area2_ = roi_split_param.split_area2(); }
  else
  { split_area2_ = split_area1_; }

  fluctuation_range_large_ = roi_split_param.fluctuation_range_large(); //this->phase_ == TRAIN
  fluctuation_range_small_ = roi_split_param.fluctuation_range_small(); //this->phase_ == TRAIN
  srand(time(0));
}

template <typename Dtype>
void ROISplitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  top[0]->Reshape(2, 1, 1, 1);

  top[1]->Reshape(bottom[0]->num(),2,1,1); //hash function for different size of ROI (num for each class, index for normal[1] small[2])

}

double generateGaussianDistribution(double mu, double sigma)
{
   const double epsilon = 1e-5;
   const double two_pi = 2.0*3.14159265359;
   static double z0,z1;
   static bool generate;
   generate = !generate;
 
   if(!generate)
      return z1*sigma+mu;

   double u1,u2;
   do
   {
      u1 = rand()*(1.0/RAND_MAX);  //RAND_MAX 32767
      u2 = rand()*(1.0/RAND_MAX);
   }
   while (u1<epsilon);

   z0 = sqrt(-2.0*log(u1))*cos(two_pi*u2);
   z1 = sqrt(-2.0*log(u1))*sin(two_pi*u2);

   return z0 * sigma + mu;
}

template <typename Dtype>
void ROISplitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  const Dtype* bottom_rois = bottom[0]->cpu_data();
  // Number of ROIs
  int num_rois = bottom[0]->num();

  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* hash_table = top[1]->mutable_cpu_data();
  //Dtype* hash_bool_table = top[2]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  caffe_set(top[1]->count(), Dtype(-FLT_MAX), hash_table);
  //caffe_set(top[2]->count(), Dtype(-FLT_MAX), hash_bool_table);

  int num_small_roi = 0;
  int num_middle_roi = 0;
  int num_large_roi = 0;
  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R

  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind = bottom_rois[0];
    CHECK_GE(roi_batch_ind, 0);
    //CHECK_LT(roi_batch_ind, batch_size);
    
    int height = (bottom_rois[4]-bottom_rois[2]+1);
    int width = (bottom_rois[3]-bottom_rois[1]+1);
 
    int final_split_area1 = split_area1_;
    int final_split_area2 = split_area2_;
    
    //generate the guassian distribution random number
    if (this->phase_ == TRAIN)
    {
       int stan_dev_large = fluctuation_range_large_/3;
       int stan_dev_small = fluctuation_range_small_/3;
 
       int num_large, num_small, num;
       bool vaild = false;  
       while(!vaild)
       {
           num_large = std::floor(generateGaussianDistribution(split_area1_,stan_dev_large)+0.5);
           num_small = std::floor(generateGaussianDistribution(split_area1_,stan_dev_small)+0.5);
           //num_large  =  split_area1_ + rand()*(1.0/RAND_MAX)*stan_dev_large;          
           //num_small  =  split_area1_ - rand()*(1.0/RAND_MAX)*stan_dev_small;          

           int choose = (1+rand()%10);
           num = choose>5?num_large:num_small;
           if(choose>5 && num_large<=split_area1_ || choose<=5 && num_small>=split_area1_)
           {
               vaild = true;
           }
       }
       
       final_split_area1 = num<=(split_area1_+fluctuation_range_small_)?(num):(split_area1_+fluctuation_range_small_);
       final_split_area1 = num>=(split_area1_-fluctuation_range_large_)?(num):(split_area1_-fluctuation_range_large_);   
    }


    //generate the guassian distribution random number
    if (this->phase_ == TRAIN)
    {
       int stan_dev_large = fluctuation_range_large_/3;
       int stan_dev_small = fluctuation_range_small_/3;
 
       int num_large, num_small, num;
       bool vaild = false;  
       while(!vaild)
       {
           num_large = std::floor(generateGaussianDistribution(split_area2_,stan_dev_large)+0.5);
           num_small = std::floor(generateGaussianDistribution(split_area2_,stan_dev_small)+0.5);
           
           int choose = (1+rand()%10);
           num = choose>5?num_large:num_small;
           if(choose>5 && num_large<=split_area2_ || choose<=5 && num_small>=split_area2_)
           {
               vaild = true;
           }
       }
       
       final_split_area2 = num<=(split_area2_+fluctuation_range_small_)?(num):(split_area2_+fluctuation_range_small_);
       final_split_area2 = num>=(split_area2_-fluctuation_range_large_)?(num):(split_area2_-fluctuation_range_large_);   
    }

    
    if (branch_num_ == 3) 
    {
       if (height * width < final_split_area1) //small RoIs
       {
          //cout<<"num_small_roi"<<num_small_roi<<endl;
          hash_table[0] = static_cast<Dtype>(num_small_roi);
          hash_table[1] = static_cast<Dtype>(1);
          num_small_roi++;
       }
       else if (height * width >= final_split_area1 && height * width <= final_split_area2) //middle RoIs
       {
          //cout<<"num_normal_roi"<<num_normal_roi<<endl;
          hash_table[0] = static_cast<Dtype>(num_middle_roi);
          hash_table[1] = static_cast<Dtype>(2);
          num_middle_roi++;
       }
       else //large RoIs
       {
          //cout<<"num_normal_roi"<<num_normal_roi<<endl;
          hash_table[0] = static_cast<Dtype>(num_large_roi);
          hash_table[1] = static_cast<Dtype>(3);
          num_large_roi++;
       }
    }
    else
    {
       if (height * width < final_split_area1) //small RoIs
       {
          //cout<<"num_small_roi"<<num_small_roi<<endl;
          hash_table[0] = static_cast<Dtype>(num_small_roi);
          hash_table[1] = static_cast<Dtype>(1);
          num_small_roi++;
       }
       else //large RoIs
       {
          //cout<<"num_normal_roi"<<num_normal_roi<<endl;
          hash_table[0] = static_cast<Dtype>(num_large_roi);
          hash_table[1] = static_cast<Dtype>(2);
          num_large_roi++;
       }
    }
    // Increment ROI data pointer
    bottom_rois += bottom[0]->offset(1);  
    hash_table += top[1]->offset(1);
    //hash_bool_table += top[2]->offset(1);
  }
  //cout<<"num_small_roi: "<<num_small_roi<<endl;
  //cout<<"num_normal_roi: "<<num_normal_roi<<endl;

  if (branch_num_ == 3)
  {
     top_data[0] = static_cast<Dtype>(num_small_roi);
     top_data += top[0]->offset(1);
     top_data[0] = static_cast<Dtype>(num_middle_roi);
     top_data += top[0]->offset(1);
     top_data[0] = static_cast<Dtype>(num_large_roi);
  }
  else
  {
     top_data[0] = static_cast<Dtype>(num_small_roi);
     top_data += top[0]->offset(1);
     top_data[0] = static_cast<Dtype>(num_large_roi);
  }
  
}

template <typename Dtype>
void ROISplitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}


#ifdef CPU_ONLY
STUB_GPU(ROISplitLayer);
#endif

INSTANTIATE_CLASS(ROISplitLayer);
REGISTER_LAYER_CLASS(ROISplit);

}  // namespace caffe
