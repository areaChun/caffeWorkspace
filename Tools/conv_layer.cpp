#include <vector>

#include "caffe/layers/conv_layer.hpp"
#include "caffe/binary.hpp"
#include <iostream> 
#include <iomanip>
#include <fstream>
#include <string> 

extern bool BINARY;
extern bool TERNARY;
using namespace std; 
namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // const Dtype* weight = this->blobs_[0]->cpu_data();
if(BINARY){
  this->blobs_[0]->binarize_data();
} 

if(TERNARY){
  this->blobs_[0]->ternarize_data(this->phase_);  //quantized from blob[0] to ternary sand stored in cpu_binary()
/*
    Dtype alpha = (Dtype) this->blobs_[0]->get_alpha();

for(int i=0; i<bottom.size(); i++){
  Blob<Dtype>* blob = bottom[i];
  caffe_cpu_scale(blob->count(), alpha, blob->cpu_data(), blob->mutable_cpu_data());
}
*/
}
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  // Output blob
  // this->blobs_[0]->ternarize_data(this->phase_);
  // caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->cpu_binary(),
  //     this->blobs_[0]->mutable_cpu_data());
  // if (this->bias_term_) {
  //   caffe_copy(this->blobs_[1]->count(), this->blobs_[1]->cpu_binary(),
  //       this->blobs_[1]->mutable_cpu_data());
  // }



  const Dtype* weight = (BINARY || TERNARY) ? this->blobs_[0]->cpu_binary() : this->blobs_[0]->cpu_data();
  // // LOG(INFO) << "Outputdebug: "<< *weight;
  // // cout << *weight;
  // string ssstr = this->layer_param_.name();
  // char * pstr=new char [ssstr.length()+1];
  // strcat(pstr,ssstr);
  this->blobs_[0]->ternarize_data(this->phase_);
  const Dtype* weight_ternarize =  this->blobs_[0]->cpu_binary() ;
  ofstream fp("/home/zcr/output_weight_ternarize.txt", ios::out | ios::app);
  //ofstream fp(pstr);
  LOG(INFO) << "Outputdebug: 写入信息";
  LOG(INFO) << "Outputdebug: "<< "num_output(0)"<<conv_param.num_output();
  LOG(INFO) << "Outputdebug: "<< "channels_"<<this->channels_;
  LOG(INFO) << "Outputdebug: "<< "kernel_size(0)"<<conv_param.kernel_size(0);
  LOG(INFO) << "Outputdebug: "<< " this->layer_param_."<< this->layer_param_.name();
  fp << this->layer_param_.name() <<endl;
  for(int i = 0; i <conv_param.num_output() * this->channels_ * conv_param.kernel_size(0) * conv_param.kernel_size(0); i++){
    fp << *(weight_ternarize+i) << " " ;
  }
  fp << " " << endl ;
  fp.close();

  ofstream original_weight("/home/zcr/output_original_weight.txt", ios::out | ios::app);
  //ofstream fp(pstr);
  LOG(INFO) << "Outputdebug: 写入信息";
  LOG(INFO) << "Outputdebug: "<< "num_output(0)"<<conv_param.num_output();
  LOG(INFO) << "Outputdebug: "<< "channels_"<<this->channels_;
  LOG(INFO) << "Outputdebug: "<< "kernel_size(0)"<<conv_param.kernel_size(0);
  LOG(INFO) << "Outputdebug: "<< " this->layer_param_."<< this->layer_param_.name();
  original_weight << this->layer_param_.name() <<endl;
  for(int i = 0; i <conv_param.num_output() * this->channels_ * conv_param.kernel_size(0) * conv_param.kernel_size(0); i++){
    original_weight << *(weight+i) << " " ;
  }
  original_weight << " " << endl ;
  original_weight.close();



  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* weight = (BINARY || TERNARY) ? this->blobs_[0]->cpu_binary() : this->blobs_[0]->cpu_data();

  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
