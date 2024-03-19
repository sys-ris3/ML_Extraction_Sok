/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

using namespace tensorflow;

template <typename T>
class ShuffleChannelOp : public OpKernel {
 public:
  explicit ShuffleChannelOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input = context->input(0);

    // Grab the weights tensor
    const Tensor& weights = context->input(1);

    // Grab the random scalar tensor
    const Tensor& rscalar = context->input(2);

    // check shapes of input, weights and rscalar
    const TensorShape& input_shape = input.shape();
    const TensorShape& weights_shape = weights.shape();
    const TensorShape& rscalar_shape = rscalar.shape();
    
    // Assume input is "NHWC"
    DCHECK_EQ(input_shape.dims(), 4);
    DCHECK_EQ(weights_shape.dims(), 1);
    DCHECK_EQ(rscalar_shape.dims(), 1);

    // input channels equals weights and scalar channels
    DCHECK_EQ(input_shape.dim_size(3), weights_shape.dim_size(0));
    DCHECK_EQ(weights_shape.dim_size(0), weights_shape.dim_size(0));

    // Create an output tensor
    Tensor* output= NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &output));

    // get the corresponding Eigen tensors for data access
    auto input_tensor = input.template tensor<T, 4>();
    auto weights_tensor = weights.flat<int32>();
    auto rscalar_tensor = rscalar.template flat<T>();
    auto output_tensor = output->template tensor<T, 4>();

    // all input element with mask tensor element.
    int idx_from;
    T scalar;
    for (int i = 0; i < output->shape().dim_size(0); i++) {
      for (int j = 0; j < output->shape().dim_size(1); j++) {
        for (int k = 0; k < output->shape().dim_size(2); k++) {
          for (int c = 0; c < output->shape().dim_size(3); c++) {
            idx_from = weights_tensor(c);
            scalar = rscalar_tensor(c); 
            output_tensor(i,j,k,c) = input_tensor(i,j,k,idx_from) * scalar;
          }
        }
      }
    }

  }
};

#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("ShuffleChannel").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ShuffleChannelOp<type>)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL
