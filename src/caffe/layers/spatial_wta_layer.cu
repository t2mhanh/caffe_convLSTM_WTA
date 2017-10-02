#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/spatial_wta_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SpatialWtaForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int wta_height,
    const int wta_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    Dtype* const top_data, int* mask, Dtype* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int ww = index % wta_width;
    const int wh = (index / wta_width) % wta_height;
    const int c = (index / wta_width / wta_height) % channels;
    const int n = index / wta_width / wta_height / channels;

    int hstart_lower = static_cast<int>(floor(static_cast<float>(
        wh - kernel_h)/stride_h) + 1)*stride_h;
    int wstart_lower = static_cast<int>(floor(static_cast<float>(
        ww - kernel_w)/stride_w) + 1)*stride_w;
    hstart_lower = max(hstart_lower,0);
    wstart_lower = max(wstart_lower,0);
    //int hstart_lower = max(ceil((wh - kernel_h)/stride_h)*stride_h,0);
    // const int hstart_lower = max(((wh - kernel_h)/stride_h+1)*stride_h,0);
    // int hstart_upper = wh;
    int hstart_upper = min(wh,wta_height - kernel_h);
    //const int wstart_lower = max(ceil((ww - kernel_w)/stride_w),0);
    // const int wstart_lower = max(((ww - kernel_w)/stride_w+1)*stride_w,0);
    // int wstart_upper = ww;
    int wstart_upper = min(ww,wta_width - kernel_w);
    for (int hstart = hstart_lower; hstart <= hstart_upper; hstart += stride_h){
      for (int wstart = wstart_lower; wstart <= wstart_upper; wstart += stride_w){
        const int hend = min(hstart + kernel_h, height);
        const int wend = min(wstart + kernel_w, width);
        Dtype maxval = -FLT_MAX;
        int maxidx = -1;
        const Dtype* const bottom_slice =
            bottom_data + (n * channels + c) * height * width;
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            // if (bottom_slice[h * width + w] > maxval) {
              if (fabs(bottom_slice[h * width + w]) > maxval) {
                maxidx = h * width + w;
                maxval = fabs(bottom_slice[maxidx]);
              // maxval = bottom_slice[maxidx];
            }
          }
        }
        maxidx += (n * channels + c) * height * width;
        if (index == maxidx){
          top_data[index] = bottom_data[index];
          if (mask) {
            mask[index] = 1;
          } else {
            top_mask[index] = 1;
          }
        }
        // -- these lines may set previous maximum values to 0
        // else {
        //   top_data[index] = 0;
        //   if (mask) {
        //     mask[index] = 0;
        //   } else {
        //     top_mask[index] = 0;
        //   }
        // }
      }
    }
  }
}

template <typename Dtype>
void SpatialWtaLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
       const vector<Blob<Dtype>*>& top) {
const Dtype* bottom_data = bottom[0]->gpu_data();
Dtype* top_data = top[0]->mutable_gpu_data();
int count = top[0]->count();
caffe_gpu_set(count, Dtype(0.), top_data);
// We'll output the mask to top[1] if it's of size >1.
const bool use_top_mask = top.size() > 1;
int* mask = NULL;
Dtype* top_mask = NULL;
if (use_top_mask) {
    top_mask = top[1]->mutable_gpu_data();
    caffe_gpu_set(count, Dtype(0.), top_mask);
} else {
    mask = max_mask_.mutable_gpu_data();
    caffe_gpu_set(count, 0 , mask);
}
  // NOLINT_NEXT_LINE(whitespace/operators)
  SpatialWtaForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom[0]->num(), channels_,
      height_, width_, wta_height_, wta_width_, kernel_h_,
      kernel_w_, stride_h_, stride_w_, top_data,
      mask, top_mask);
CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void SpatialWtaBackward(const int nthreads, const Dtype* const top_diff,
    const int* const mask, const Dtype* const top_mask, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    if (mask) {
      bottom_diff[index] = top_diff[index] * mask[index];
    } else {
      bottom_diff[index] = top_diff[index] * top_mask[index];
          }
        }
      }

template <typename Dtype>
void SpatialWtaLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    // We'll output the mask to top[1] if it's of size >1.
    const bool use_top_mask = top.size() > 1;
    const int* mask = NULL;
    const Dtype* top_mask = NULL;
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_mask_.gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    SpatialWtaBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask,top_mask, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}



// template <typename Dtype>
// __global__ void kernel_channel_max(const int num, const int channels,
//     const int height, const int width , const Dtype* data, Dtype* out) {
//   CUDA_KERNEL_LOOP(index, num * channels) {
//     const int c = index % channels;
//     const int n = index / channels;
//     Dtype maxval = -FLT_MAX;
//     for (int h = 0; h < height; ++h) {
//           for (int w = 0; w < width; ++w) {
//             maxval = max(data[h * width + w], maxval);
//           }
//     }
//     out[index] = maxval;
//   }
// }
//
// template <typename Dtype>
// __global__ void kernel_wta_forward(const int num, const int channels,
//     const int height, const int width , const Dtype* bottom_data, const Dtype* channel_max, Dtype* top_data) {
//   CUDA_KERNEL_LOOP(index, num) {
//     const int c = index / width / height % channels;
//     const int n = index / width / height/ channels;
//
//     top_data[index] = (bottom_data[index] == channel_max[n*c]) ? bottom_data[index] : 0;
//   }
// }
//
// template <typename Dtype>
// __global__ void kernel_wta_backward(const int num, const int channels,
//     const int height, const int width , const Dtype* bottom_data,
//     const Dtype* channel_max, const Dtype* const top_diff,
//     Dtype* bottom_diff) {
//   CUDA_KERNEL_LOOP(index, num) {
//     const int c = index / width / height % channels;
//     const int n = index / width / height/ channels;
//
//     bottom_diff[index] = (bottom_data[index] == channel_max[n*c]) ? top_diff[index] : 0;
//   }
// }
//
// (const int num, const int channels,
//     const int height, const int width , const Dtype* data, Dtype* out)
//
// template <typename Dtype>
// void SpatialConvWtaLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//     const vector<Blob<Dtype>*>& top) {
//   const Dtype* bottom_data = bottom[0]->gpu_data();
//   Dtype* top_data = top[0]->mutable_gpu_data();
//   //const int count = bottom[0]->count();
//   const int num_data_slice = bottom[0]->num() * channels_;
//   // NOLINT_NEXT_LINE(whitespace/operators)
//   kernel_wta_forward<Dtype><<<CAFFE_GET_BLOCKS(num_data_slice), CAFFE_CUDA_NUM_THREADS>>>(
//     bottom[0]->num(), channels_, height_, width_, bottom_data, channel_max_data);
//   CUDA_POST_KERNEL_CHECK;
//   // << " count: " << count << " bottom_data: "
//   //     << (unsigned long)bottom_data
//   //     << " top_data: " << (unsigned long)top_data
//   //     << " blocks: " << CAFFE_GET_BLOCKS(count)
//   //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
// }
// template <typename Dtype>
// void SpatialConvWtaLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
//     const vector<bool>& propagate_down,
//     const vector<Blob<Dtype>*>& bottom) {
//   if (propagate_down[0]) {
//     const Dtype* bottom_data = bottom[0]->gpu_data();
//     const Dtype* top_diff = top[0]->gpu_diff();
//     Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
//     const int count = bottom[0]->count();
//     const int num_data_slice = bottom[0]->num() * channels_;
//     // NOLINT_NEXT_LINE(whitespace/operators)
//     SpatialConvWtaBackward<Dtype><<<CAFFE_GET_BLOCKS(num_data_slice), CAFFE_CUDA_NUM_THREADS>>>(
//         num_data_slice, top_diff, bottom[0]->num(), channels_, height_, width_, bottom_diff, bottom_data);
//     CUDA_POST_KERNEL_CHECK;
//   }
// }



INSTANTIATE_LAYER_GPU_FUNCS(SpatialWtaLayer);


}  // namespace caffe
