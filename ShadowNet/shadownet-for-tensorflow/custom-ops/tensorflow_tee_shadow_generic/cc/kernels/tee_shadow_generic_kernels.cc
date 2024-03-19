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
#include "third_party/darknet/tee_shadow_net.h"

using namespace tensorflow;

template <typename T>
class TeeShadowGenericOp : public OpKernel {
 public:
  explicit TeeShadowGenericOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context,
              context->GetAttr("h", &h_));
      OP_REQUIRES_OK(context,
              context->GetAttr("w", &w_));
      OP_REQUIRES_OK(context,
              context->GetAttr("c", &c_));
      OP_REQUIRES_OK(context,
              context->GetAttr("pos", &position_));
      OP_REQUIRES(context, h_ >= 0,
              errors::InvalidArgument("height out of range"));
      OP_REQUIRES(context, w_ >= 0,
              errors::InvalidArgument("width out of range"));
      OP_REQUIRES(context, c_ >= 0,
              errors::InvalidArgument("channel out of range"));
      // TODO add further checks make sure position_ is in a list ("conv1", ..., "results")
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const TensorShape& input_shape = input_tensor.shape();

    // check input is "NHWC" 
    DCHECK_EQ(input_shape.dims(),4);

    // create output shape
    TensorShape output_shape;

    // TODO: Assume the last layer is always named "results"
    if(position_.compare("results") == 0) { // results shape(batch, channels)
      CHECK_GT(c_, 0);
      output_shape.AddDim(input_shape.dim_size(0));
      output_shape.AddDim(c_);
    } else {
      // TODO Assume NHWC
      output_shape.AddDim(input_shape.dim_size(0));
      output_shape.AddDim(h_);
      output_shape.AddDim(w_);
      output_shape.AddDim(c_);
    }
    
    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));

    // get the corresponding Eigen tensors for data access
    const void *input_flat = input_tensor.flat<T>().data();
    void *output_flat = output_tensor->flat<T>().data();
    

    const char *shadow_pos = position_.c_str();
    size_t input_size = input_tensor.AllocatedBytes();
    size_t output_size = output_tensor->AllocatedBytes();
    darknet_predict(shadow_pos,input_size, input_flat, output_size,output_flat);
    return;
  }

 private:
  int h_;
  int w_;
  int c_;
  string position_;
};

#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("TeeShadowGeneric").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      TeeShadowGenericOp<type>)

//TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);
REGISTER_KERNEL(float);
#undef REGISTER_KERNEL
