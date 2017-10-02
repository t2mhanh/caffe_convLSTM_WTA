#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/spatial_wta_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

// NEED LayerSetUp / Reshape OR NOT ??????? WHEN NEED???

template <typename Dtype>
void SpatialWtaLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  SpatialWtaParameter spatial_wta_param = this->layer_param_.spatial_wta_param();
  if (spatial_wta_param.global_wta()) {
    CHECK(!(spatial_wta_param.has_kernel_size() ||
      spatial_wta_param.has_kernel_h() || spatial_wta_param.has_kernel_w()))
      << "With Global_WTA: true Filter size cannot specified";
  } else {
    CHECK(!spatial_wta_param.has_kernel_size() !=
      !(spatial_wta_param.has_kernel_h() && spatial_wta_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
    CHECK(spatial_wta_param.has_kernel_size() ||
      (spatial_wta_param.has_kernel_h() && spatial_wta_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
    }
  CHECK((!spatial_wta_param.has_stride() && spatial_wta_param.has_stride_h()
      && spatial_wta_param.has_stride_w())
      || (!spatial_wta_param.has_stride_h() && !spatial_wta_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";

      global_wta_ = spatial_wta_param.global_wta();
      if (global_wta_) {
        kernel_h_ = bottom[0]->height();
        kernel_w_ = bottom[0]->width();
      } else {
    if (spatial_wta_param.has_kernel_size()) {
      kernel_h_ = kernel_w_ = spatial_wta_param.kernel_size();
    } else {
      kernel_h_ = spatial_wta_param.kernel_h();
      kernel_w_ = spatial_wta_param.kernel_w();
    }
}
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";

  if (!spatial_wta_param.has_stride_h()) {
    stride_h_ = stride_w_ = spatial_wta_param.stride();
  } else {
    stride_h_ = spatial_wta_param.stride_h();
    stride_w_ = spatial_wta_param.stride_w();
  }
  if (global_wta_) {
    CHECK(stride_h_ == 1 && stride_w_ == 1)
      << "With Global_WTA: true; stride = 1";
  }
}

template <typename Dtype>
void SpatialWtaLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  if (global_wta_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  }
  wta_height_ = static_cast<int>(height_);
  wta_width_ = static_cast<int>(width_);

  top[0]->Reshape(bottom[0]->num(), channels_, wta_height_,
      wta_width_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  if (top.size() == 1) {
    max_mask_.Reshape(bottom[0]->num(), channels_, wta_height_,
        wta_width_);
  }
}

template <typename Dtype>
void SpatialWtaLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const int channels_ = bottom[0]->channels();
  const int height_ = bottom[0]->height();
  const int width_ = bottom[0]->width();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  if (use_top_mask) {
    top_mask = top[1]->mutable_cpu_data();
    caffe_set(top_count, Dtype(0), top_mask);
  } else {
    mask = max_mask_.mutable_cpu_data();
    caffe_set(top_count, 0, mask);
  }
  int n_block_height = static_cast<int>(floor(static_cast<float>(
      height_ - kernel_h_) / stride_h_)) + 1;
  int n_block_width = static_cast<int>(floor(static_cast<float>(
      width_ - kernel_w_) / stride_w_)) + 1;
  //caffe_set(top_count, Dtype(-FLT_MAX), top_data); //for method1
  caffe_set(top_count,Dtype(0.),top_data);
  //const int count = bottom[0]->count();
  for (int n = 0; n < bottom[0]->num(); ++n){
    for (int c = 0; c < channels_; ++c){
      for (int wh = 0; wh < n_block_height; ++wh){
        for (int ww = 0; ww < n_block_width; ++ww){
          int hstart = wh * stride_h_;
          int wstart = ww * stride_w_;
          int hend = min(hstart + kernel_h_, height_);
          int wend = min(wstart + kernel_w_, width_);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          int max_idx = -1;
          Dtype max_val = -FLT_MAX;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = h * width_ + w;
              // if (bottom_data[index] > max_val) {
              //   max_val= bottom_data[index];
                if (fabs(bottom_data[index]) > max_val) {
                  max_val= fabs(bottom_data[index]);
                  max_idx = index;
              }
            }
          }
          top_data[max_idx] = bottom_data[max_idx];
          if (use_top_mask) {
            top_mask[max_idx] = static_cast<int>(1);
          } else {
            mask[max_idx] = 1;
          }
        }
      }
      // compute offset
      bottom_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
      if (use_top_mask) {
        top_mask += top[0]->offset(0, 1);
      } else {
        mask += top[0]->offset(0, 1);
      }
    }
  }
}

template <typename Dtype>
void SpatialWtaLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
    // The main loop
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_mask_.cpu_data();
    }
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int h = 0; h < wta_height_; ++h) {
          for (int w = 0; w < wta_width_; ++w) {
            const int index = h * width_ + w;
            const int mask_index =
                use_top_mask ? top_mask[index] : mask[index];
            bottom_diff[index] = top_diff[index] * mask_index;
          }
        }
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
}
#ifdef CPU_ONLY
STUB_GPU(SpatialWtaLayer);
#endif

INSTANTIATE_CLASS(SpatialWtaLayer);
REGISTER_LAYER_CLASS(SpatialWta);
}  // namespace caffe
