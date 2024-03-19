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
#include "tensorflow/core/framework/tensor_shape.h"

using namespace tensorflow;

template <typename T>
class LinearTransformGenericOp : public OpKernel {
 public:
  explicit LinearTransformGenericOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Grab the weights tensor
    const Tensor& weights_tensor = context->input(1);

    // Grab the random scalar tensor
    const Tensor& rscalar_tensor = context->input(2);

    // Grab the random scalar tensor
    const Tensor& bias_tensor = context->input(3);

    // check shapes of input, weights and rscalar
    const TensorShape& input_shape = input_tensor.shape();
    const TensorShape& weights_shape = weights_tensor.shape();
    const TensorShape& rscalar_shape = rscalar_tensor.shape();
    const TensorShape& bias_shape = bias_tensor.shape();
    
    // check input is "NHWC" 
    DCHECK_EQ(input_shape.dims(),4);
	// check rscalar is [r1, ..., rn]
    DCHECK_EQ(rscalar_shape.dims(), 1);
    DCHECK_EQ(bias_shape.dims(), 1);
	// check weights is [[w11, ..., w1n],[w21,...,w2n]]
    DCHECK_EQ(weights_shape.dims(), 2);
    DCHECK_EQ(weights_shape.dim_size(1), rscalar_shape.dim_size(0));
    DCHECK_EQ(weights_shape.dim_size(1), bias_shape.dim_size(0));
    
   // create output shape
    TensorShape output_shape;
    output_shape.AddDim(input_shape.dim_size(0));
    output_shape.AddDim(input_shape.dim_size(1));
    output_shape.AddDim(input_shape.dim_size(2));
    output_shape.AddDim(weights_shape.dim_size(1));


    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));

    // get the corresponding Eigen tensors for data access
    auto input= input_tensor.tensor<T, 4>();
    auto weights= weights_tensor.tensor<int32, 2>();
    auto rscalar = rscalar_tensor.tensor<T,1>();
    auto bias = bias_tensor.tensor<T,1>();
    auto output = output_tensor->tensor<T, 4>();

	int idx_from, idx_rand;
	T scalar;
    for (int i = 0; i < output_tensor->shape().dim_size(0); i++) {
      for (int j = 0; j < output_tensor->shape().dim_size(1); j++) {
      	for (int k = 0; k < output_tensor->shape().dim_size(2); k++) {
      	  for (int n = 0; n < output_tensor->shape().dim_size(3); n++) {
            idx_from = weights(0,n);
            idx_rand = weights(1,n);
            scalar = rscalar(n);

        	output(i,j,k,n) = 
               input(i, j, k, idx_from) * scalar + input(i,j,k,idx_rand) + bias(n);
		  }
		}
      }
    }

  }
};

#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("LinearTransformGeneric").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      LinearTransformGenericOp<type>)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
