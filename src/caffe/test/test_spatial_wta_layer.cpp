#include <algorithm>
#include <vector>
#include <iostream> // Hanh add for std::cout
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/spatial_wta_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SpatialWtaLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SpatialWtaLayerTest()
  : blob_bottom_(new Blob<Dtype>()),
    blob_top_(new Blob<Dtype>()),
    blob_top_mask_(new Blob<Dtype>()) {}
  virtual void SetUp() {
      Caffe::set_random_seed(1701);
      blob_bottom_->Reshape(2, 3 , 6 , 5);
      // fill the values
      FillerParameter filler_param;
      GaussianFiller<Dtype> filler(filler_param);
      filler.Fill(this->blob_bottom_);
      blob_bottom_vec_.push_back(blob_bottom_);
      blob_top_vec_.push_back(blob_top_);
    }
  virtual ~SpatialWtaLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_top_mask_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_mask_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  // Test
  void TestForwardGlobal() {
    LayerParameter layer_param;
    SpatialWtaParameter* spatial_wta_param = layer_param.mutable_spatial_wta_param();
    //spatial_wta_layer_param->set_kernel_size(2);
    spatial_wta_param->set_global_wta(true);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 3, 5);
    // Input: 2x 2 channels of:
    //     [1 2 5 2 3]
    //     [3 4 7 4 2]
    //     [1 2 5 2 3]
    for (int i = 0; i < 15 * num * channels; i += 15) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 1;
      blob_bottom_->mutable_cpu_data()[i +  1] = 2;
      blob_bottom_->mutable_cpu_data()[i +  2] = 5;
      blob_bottom_->mutable_cpu_data()[i +  3] = 2;
      blob_bottom_->mutable_cpu_data()[i +  4] = 3;
      blob_bottom_->mutable_cpu_data()[i +  5] = 3;
      blob_bottom_->mutable_cpu_data()[i +  6] = 4;
      blob_bottom_->mutable_cpu_data()[i +  7] = -7;
      blob_bottom_->mutable_cpu_data()[i +  8] = 4;
      blob_bottom_->mutable_cpu_data()[i +  9] = 2;
      blob_bottom_->mutable_cpu_data()[i + 10] = 1;
      blob_bottom_->mutable_cpu_data()[i + 11] = 2;
      blob_bottom_->mutable_cpu_data()[i + 12] = 5;
      blob_bottom_->mutable_cpu_data()[i + 13] = 2;
      blob_bottom_->mutable_cpu_data()[i + 14] = 3;
    }
    SpatialWtaLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 3);
    EXPECT_EQ(blob_top_->width(), 5);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->num(), num);
      EXPECT_EQ(blob_top_mask_->channels(), channels);
      EXPECT_EQ(blob_top_mask_->height(), 3);
      EXPECT_EQ(blob_top_mask_->width(), 5);
    }
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 2x 2 channels of:
    //     [0 0 0 0 0]
    //     [0 0 7 0 0]
    //     [0 0 0 0 0]
    for (int i = 0; i < 15 * num * channels; i += 15) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 3], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 6], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7], -7);
      EXPECT_EQ(blob_top_->cpu_data()[i + 8], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 9], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 0);
    }
    if (blob_top_vec_.size() > 1) {
      //     [0 0 0 0 0]
      //     [0 0 1 0 0]
      //     [0 0 0 0 0]
      for (int i = 0; i < 15 * num * channels; i += 15) {
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  0],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  1],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  2],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  3],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  4],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  5],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  6], 0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  7], 1);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  8], 0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  9], 0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 10], 0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 11], 0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 12], 0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 13], 0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 14], 0);
      }
    }
  }

  void TestForwardSq() {
    LayerParameter layer_param;
    SpatialWtaParameter* spatial_wta_param = layer_param.mutable_spatial_wta_param();
    spatial_wta_param->set_kernel_size(2);
    // spatial_wta_param->set_global_wta(true);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 3, 3 );
    // Input: 2x 2 channels of:
    //     [1 2 5]
    //     [3 4 7]
    //     [1 4 6]
    for (int i = 0; i < 9 * num * channels; i += 9) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 1;
      blob_bottom_->mutable_cpu_data()[i +  1] = 2;
      blob_bottom_->mutable_cpu_data()[i +  2] = 5;
      blob_bottom_->mutable_cpu_data()[i +  3] = 3;
      blob_bottom_->mutable_cpu_data()[i +  4] = 4;
      blob_bottom_->mutable_cpu_data()[i +  5] = 7;
      blob_bottom_->mutable_cpu_data()[i +  6] = 1;
      blob_bottom_->mutable_cpu_data()[i +  7] = 4;
      blob_bottom_->mutable_cpu_data()[i +  8] = 6;
    }
    // EXPECT_EQ(blob_bottom_->mutable_cpu_data()[3], 1);
    SpatialWtaLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 3);
    EXPECT_EQ(blob_top_->width(), 3);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->num(), num);
      EXPECT_EQ(blob_top_mask_->channels(), channels);
      EXPECT_EQ(blob_top_mask_->height(), 3);
      EXPECT_EQ(blob_top_mask_->width(), 3);
    }
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 2x 2 channels of:
    //     [0 0 0 ]
    //     [0 4 7 ]
    //     [0 0 0 ]
    for (int i = 0; i < 9 * num * channels; i += 9) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 3], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4], 4);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5], 7);
      EXPECT_EQ(blob_top_->cpu_data()[i + 6], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 8], 0);
    }
    if (blob_top_vec_.size() > 1) {
    //     //     [0 0 0 ]
    //     //     [0 1 1 ]
    //     //     [0 0 0]
    for (int i = 0; i < 9 * num * channels; i += 9) {
      EXPECT_EQ(blob_top_mask_->cpu_data()[i +  0],  0);
      EXPECT_EQ(blob_top_mask_->cpu_data()[i +  1],  0);
      EXPECT_EQ(blob_top_mask_->cpu_data()[i +  2],  0);
      EXPECT_EQ(blob_top_mask_->cpu_data()[i +  3],  0);
      EXPECT_EQ(blob_top_mask_->cpu_data()[i +  4],  1);
      EXPECT_EQ(blob_top_mask_->cpu_data()[i +  5],  1);
      EXPECT_EQ(blob_top_mask_->cpu_data()[i +  6], 0);
      EXPECT_EQ(blob_top_mask_->cpu_data()[i +  7], 0);
      EXPECT_EQ(blob_top_mask_->cpu_data()[i +  8], 0);
      }
    }
  }

  // void TestForwardSq() {
  //   LayerParameter layer_param;
  //   SpatialWtaParameter* spatial_wta_param = layer_param.mutable_spatial_wta_param();
  //   spatial_wta_param->set_kernel_size(2);
  //   const int num = 2;
  //   const int channels = 2;
  //   blob_bottom_->Reshape(num, channels, 3, 5);
  //   // Input: 2x 2 channels of:
  //   //     [1 2 5 2 3]
  //   //     [3 4 7 4 2]
  //   //     [1 2 5 2 3]
  //   for (int i = 0; i < 15 * num * channels; i += 15) {
  //     blob_bottom_->mutable_cpu_data()[i +  0] = 1;
  //     blob_bottom_->mutable_cpu_data()[i +  1] = 2;
  //     blob_bottom_->mutable_cpu_data()[i +  2] = 5;
  //     blob_bottom_->mutable_cpu_data()[i +  3] = 2;
  //     blob_bottom_->mutable_cpu_data()[i +  4] = 3;
  //     blob_bottom_->mutable_cpu_data()[i +  5] = 3;
  //     blob_bottom_->mutable_cpu_data()[i +  6] = 4;
  //     blob_bottom_->mutable_cpu_data()[i +  7] = 7;
  //     blob_bottom_->mutable_cpu_data()[i +  8] = 4;
  //     blob_bottom_->mutable_cpu_data()[i +  9] = 2;
  //     blob_bottom_->mutable_cpu_data()[i + 10] = 1;
  //     blob_bottom_->mutable_cpu_data()[i + 11] = 2;
  //     blob_bottom_->mutable_cpu_data()[i + 12] = 5;
  //     blob_bottom_->mutable_cpu_data()[i + 13] = 2;
  //     blob_bottom_->mutable_cpu_data()[i + 14] = 3;
  //   }
  //   SpatialWtaLayer<Dtype> layer(layer_param);
  //   layer.SetUp(blob_bottom_vec_, blob_top_vec_);
  //   EXPECT_EQ(blob_top_->num(), num);
  //   EXPECT_EQ(blob_top_->channels(), channels);
  //   EXPECT_EQ(blob_top_->height(), 3);
  //   EXPECT_EQ(blob_top_->width(), 5);
  //   if (blob_top_vec_.size() > 1) {
  //     EXPECT_EQ(blob_top_mask_->num(), num);
  //     EXPECT_EQ(blob_top_mask_->channels(), channels);
  //     EXPECT_EQ(blob_top_mask_->height(), 3);
  //     EXPECT_EQ(blob_top_mask_->width(), 5);
  //   }
  //   layer.Forward(blob_bottom_vec_, blob_top_vec_);
  //   // Expected output: 2x 2 channels of:
  //   //     [0 0 0 0 0]
  //   //     [0 4 7 4 0]
  //   //     [0 0 0 0 0]
  //   for (int i = 0; i < 15 * num * channels; i += 15) {
  //     EXPECT_EQ(blob_top_->cpu_data()[i + 0], 0);
  //     EXPECT_EQ(blob_top_->cpu_data()[i + 1], 0);
  //     EXPECT_EQ(blob_top_->cpu_data()[i + 2], 0);
  //     EXPECT_EQ(blob_top_->cpu_data()[i + 3], 0);
  //     EXPECT_EQ(blob_top_->cpu_data()[i + 4], 0);
  //     EXPECT_EQ(blob_top_->cpu_data()[i + 5], 0);
  //     EXPECT_EQ(blob_top_->cpu_data()[i + 6], 4);
  //     EXPECT_EQ(blob_top_->cpu_data()[i + 7], 7);
  //     EXPECT_EQ(blob_top_->cpu_data()[i + 8], 4);
  //     EXPECT_EQ(blob_top_->cpu_data()[i + 9], 0);
  //     EXPECT_EQ(blob_top_->cpu_data()[i + 10], 0);
  //     EXPECT_EQ(blob_top_->cpu_data()[i + 11], 0);
  //     EXPECT_EQ(blob_top_->cpu_data()[i + 12], 0);
  //     EXPECT_EQ(blob_top_->cpu_data()[i + 13], 0);
  //     EXPECT_EQ(blob_top_->cpu_data()[i + 14], 0);
  //   }
  //   if (blob_top_vec_.size() > 1) {
  //     //     [0 0 0 0 0]
  //     //     [0 1 1 1 0]
  //     //     [0 0 0 0 0]
  //     for (int i = 0; i < 15 * num * channels; i += 15) {
  //       EXPECT_EQ(blob_top_mask_->cpu_data()[i +  0],  0);
  //       EXPECT_EQ(blob_top_mask_->cpu_data()[i +  1],  0);
  //       EXPECT_EQ(blob_top_mask_->cpu_data()[i +  2],  0);
  //       EXPECT_EQ(blob_top_mask_->cpu_data()[i +  3],  0);
  //       EXPECT_EQ(blob_top_mask_->cpu_data()[i +  4],  0);
  //       EXPECT_EQ(blob_top_mask_->cpu_data()[i +  5],  0);
  //       EXPECT_EQ(blob_top_mask_->cpu_data()[i +  6], 1);
  //       EXPECT_EQ(blob_top_mask_->cpu_data()[i +  7], 1);
  //       EXPECT_EQ(blob_top_mask_->cpu_data()[i +  8], 1);
  //       EXPECT_EQ(blob_top_mask_->cpu_data()[i +  9], 0);
  //       EXPECT_EQ(blob_top_mask_->cpu_data()[i + 10], 0);
  //       EXPECT_EQ(blob_top_mask_->cpu_data()[i + 11], 0);
  //       EXPECT_EQ(blob_top_mask_->cpu_data()[i + 12], 0);
  //       EXPECT_EQ(blob_top_mask_->cpu_data()[i + 13], 0);
  //       EXPECT_EQ(blob_top_mask_->cpu_data()[i + 14], 0);
  //     }
  //   }
  // }

  void TestForwardRectH() {
    LayerParameter layer_param;
    SpatialWtaParameter* spatial_wta_param = layer_param.mutable_spatial_wta_param();
    spatial_wta_param->set_kernel_h(3);
    spatial_wta_param->set_kernel_w(2);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 3, 5);
    // Input: 2x 2 channels of:
    //     [1 2 5 2 3]
    //     [3 4 7 4 2]
    //     [1 2 5 2 3]
    for (int i = 0; i < 15 * num * channels; i += 15) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 1;
      blob_bottom_->mutable_cpu_data()[i +  1] = 2;
      blob_bottom_->mutable_cpu_data()[i +  2] = 5;
      blob_bottom_->mutable_cpu_data()[i +  3] = 2;
      blob_bottom_->mutable_cpu_data()[i +  4] = 3;
      blob_bottom_->mutable_cpu_data()[i +  5] = 3;
      blob_bottom_->mutable_cpu_data()[i +  6] = 4;
      blob_bottom_->mutable_cpu_data()[i +  7] = 7;
      blob_bottom_->mutable_cpu_data()[i +  8] = 4;
      blob_bottom_->mutable_cpu_data()[i +  9] = 2;
      blob_bottom_->mutable_cpu_data()[i + 10] = 1;
      blob_bottom_->mutable_cpu_data()[i + 11] = 2;
      blob_bottom_->mutable_cpu_data()[i + 12] = 5;
      blob_bottom_->mutable_cpu_data()[i + 13] = 2;
      blob_bottom_->mutable_cpu_data()[i + 14] = 3;
    }
    SpatialWtaLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 3);
    EXPECT_EQ(blob_top_->width(), 5);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->num(), num);
      EXPECT_EQ(blob_top_mask_->channels(), channels);
      EXPECT_EQ(blob_top_mask_->height(), 3);
      EXPECT_EQ(blob_top_mask_->width(), 5);
    }
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 2x 2 channels of:
    //     [0 0 0 0 0]
    //     [0 4 7 4 0]
    //     [0 0 0 0 0]
    for (int i = 0; i < 15 * num * channels; i += 15) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 3], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 6], 4);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7], 7);
      EXPECT_EQ(blob_top_->cpu_data()[i + 8], 4);
      EXPECT_EQ(blob_top_->cpu_data()[i + 9], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 0);
    }
    if (blob_top_vec_.size() > 1) {
      //     [0 0 0 0 0]
      //     [0 1 1 1 0]
      //     [0 0 0 0 0]
      for (int i = 0; i < 15 * num * channels; i += 15) {
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  0],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  1],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  2],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  3],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  4],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  5],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  6], 1);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  7], 1);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  8], 1);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  9], 0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 10], 0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 11], 0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 12], 0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 13], 0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 14], 0);
      }
    }
  }
  void TestForwardRectW() {
    LayerParameter layer_param;
    SpatialWtaParameter* spatial_wta_param = layer_param.mutable_spatial_wta_param();
    spatial_wta_param->set_kernel_h(2);
    spatial_wta_param->set_kernel_w(3);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 3, 5);
    // Input: 2x 2 channels of:
    //     [1 2 5 2 3]
    //     [3 4 7 4 2]
    //     [1 2 5 2 3]
    for (int i = 0; i < 15 * num * channels; i += 15) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 1;
      blob_bottom_->mutable_cpu_data()[i +  1] = 2;
      blob_bottom_->mutable_cpu_data()[i +  2] = 5;
      blob_bottom_->mutable_cpu_data()[i +  3] = 2;
      blob_bottom_->mutable_cpu_data()[i +  4] = 3;
      blob_bottom_->mutable_cpu_data()[i +  5] = 3;
      blob_bottom_->mutable_cpu_data()[i +  6] = 4;
      blob_bottom_->mutable_cpu_data()[i +  7] = 7;
      blob_bottom_->mutable_cpu_data()[i +  8] = 4;
      blob_bottom_->mutable_cpu_data()[i +  9] = 2;
      blob_bottom_->mutable_cpu_data()[i + 10] = 1;
      blob_bottom_->mutable_cpu_data()[i + 11] = 2;
      blob_bottom_->mutable_cpu_data()[i + 12] = 5;
      blob_bottom_->mutable_cpu_data()[i + 13] = 2;
      blob_bottom_->mutable_cpu_data()[i + 14] = 3;
    }
    SpatialWtaLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 3);
    EXPECT_EQ(blob_top_->width(), 5);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->num(), num);
      EXPECT_EQ(blob_top_mask_->channels(), channels);
      EXPECT_EQ(blob_top_mask_->height(), 3);
      EXPECT_EQ(blob_top_mask_->width(), 5);
    }
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 2x 2 channels of:
    //     [0 0 0 0 0]
    //     [0 0 7 0 0]
    //     [0 0 0 0 0]
    for (int i = 0; i < 15 * num * channels; i += 15) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 3], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 6], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7], 7);
      EXPECT_EQ(blob_top_->cpu_data()[i + 8], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 9], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 0);
    }
    if (blob_top_vec_.size() > 1) {
      //     [0 0 0 0 0]
      //     [0 0 1 0 0]
      //     [0 0 0 0 0]
      for (int i = 0; i < 15 * num * channels; i += 15) {
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  0],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  1],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  2],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  3],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  4],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  5],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  6], 0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  7], 1);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  8], 0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  9], 0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 10], 0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 11], 0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 12], 0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 13], 0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 14], 0);
      }
    }
  }
  // THIS TEST DOESN'T WORK BECAUSE I THINK GRADIENT CHECK ISN'T SUITABLE
  //FOR THIS LAYER, (ReLU layer as well):
  // void TestBackward(){
  //   LayerParameter layer_param;
  //   SpatialWtaLayer<Dtype> layer(layer_param);
  //   GradientChecker<Dtype> checker(1e-4, 1e-2);
  //   checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
  //       this->blob_top_vec_);
  // }
};
  TYPED_TEST_CASE(SpatialWtaLayerTest, TestDtypesAndDevices);
  TYPED_TEST(SpatialWtaLayerTest, TestSetup) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    SpatialWtaParameter* spatial_wta_param = layer_param.mutable_spatial_wta_param();
    spatial_wta_param->set_kernel_size(3);
    spatial_wta_param->set_stride(2);
    SpatialWtaLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
    EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
    EXPECT_EQ(this->blob_top_->height(), 6);
    EXPECT_EQ(this->blob_top_->width(), 5);
  }
  TYPED_TEST(SpatialWtaLayerTest, TestSetupGlobalWta) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    SpatialWtaParameter* spatial_wta_param = layer_param.mutable_spatial_wta_param();
    spatial_wta_param->set_global_wta(true);
    SpatialWtaLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
    EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
    EXPECT_EQ(this->blob_top_->height(), 6);
    EXPECT_EQ(this->blob_top_->width(), 5);
  }
  TYPED_TEST(SpatialWtaLayerTest, TestForwardWta) {
    this->TestForwardGlobal();
    this->TestForwardSq();
    this->TestForwardRectH();
    this->TestForwardRectW();
  }
  TYPED_TEST(SpatialWtaLayerTest, TestForwardWtaTopMask) {
    this->blob_top_vec_.push_back(this->blob_top_mask_);
    this->TestForwardGlobal();
    this->TestForwardSq();
    this->TestForwardRectH();
    this->TestForwardRectW();
  }
  TYPED_TEST(SpatialWtaLayerTest, TestGradient) {
    typedef typename TypeParam::Dtype Dtype;
    for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
      for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
        LayerParameter layer_param;
        SpatialWtaParameter* spatial_wta_param = layer_param.mutable_spatial_wta_param();
        spatial_wta_param->set_kernel_h(kernel_h);
        spatial_wta_param->set_kernel_w(kernel_w);
        spatial_wta_param->set_stride(2);
        spatial_wta_param->set_pad(1);
        SpatialWtaLayer<Dtype> layer(layer_param);
        GradientChecker<Dtype> checker(1e-4, 1e-2);
        checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
            this->blob_top_vec_);
      }
    }
  }

  TYPED_TEST(SpatialWtaLayerTest, TestGradientTopMask) {
    typedef typename TypeParam::Dtype Dtype;
    for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
      for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
        LayerParameter layer_param;
        SpatialWtaParameter* spatial_wta_param = layer_param.mutable_spatial_wta_param();
        spatial_wta_param->set_kernel_h(kernel_h);
        spatial_wta_param->set_kernel_w(kernel_w);
        spatial_wta_param->set_stride(2);
        this->blob_top_vec_.push_back(this->blob_top_mask_);
        SpatialWtaLayer<Dtype> layer(layer_param);
        GradientChecker<Dtype> checker(1e-4, 1e-2);
        checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
            this->blob_top_vec_);
        this->blob_top_vec_.pop_back();
      }
    }
  }

}  // namespace caffe
