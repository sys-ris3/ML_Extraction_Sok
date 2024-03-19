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
class AddMaskOp : public OpKernel {
 public:
  explicit AddMaskOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Grab the weights tensor
    const Tensor& weights_tensor = context->input(1);

    // Grab the random scalar tensor
    const Tensor& rscalar_tensor = context->input(2);

    // check shapes of input, weights and rscalar
    const TensorShape& input_shape = input_tensor.shape();
    const TensorShape& weights_shape = weights_tensor.shape();
    const TensorShape& rscalar_shape = rscalar_tensor.shape();
    
    // check input is a standing vector
    DCHECK_EQ(input_shape.dims(), weights_shape.dims());
    DCHECK_EQ(rscalar_shape.dims(), 0);
    int dims = input_shape.dims();
    for (int i = 0; i < dims; i++) {
	// debug build check
    	DCHECK_EQ(input_shape.dim_size(i), weights_shape.dim_size(i));
	// release build check
    	// CHECK_EQ(input_shape.dim_size(i), weights_shape.dim_size(i));
    }

    auto input = input_tensor.flat<T>();
    auto weights = weights_tensor.flat<T>();
    auto rscalar = rscalar_tensor.flat<T>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->template flat<T>();

    // Mask all input element with mask tensor element.
    const int N = input.size();
    for (int i = 0; i < N; i++) {
      output_flat(i) = input(i) + rscalar(0) * weights(i);
    }

  }
};

#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("AddMask").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      AddMaskOp<type>)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL
