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
class TeeShadowOp : public OpKernel {
 public:
  explicit TeeShadowOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context,
              context->GetAttr("units", &units_));
      OP_REQUIRES_OK(context,
              context->GetAttr("pos", &position_));
      OP_REQUIRES(context, units_ >= 0,
              errors::InvalidArgument("units out of range"));
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

    if(position_.compare("results") == 0) { // results shape(batch, channels)
      CHECK_GT(units_, 0);
      output_shape.AddDim(input_shape.dim_size(0));
      output_shape.AddDim(units_);
    } else if(position_.compare(0, 2, "pw") == 0 ||
        position_.compare(0, 4, "conv") == 0) { 
      CHECK_GT(units_, 0);
      output_shape.AddDim(input_shape.dim_size(0));
      if (position_.compare(0, 8, "pwconv13") == 0) { // handle avgpool
        output_shape.AddDim(1);
        output_shape.AddDim(1);
      } else {
        output_shape.AddDim(input_shape.dim_size(1));
        output_shape.AddDim(input_shape.dim_size(2));
      }
      output_shape.AddDim(units_);
    } else {
      output_shape.AddDim(input_shape.dim_size(0));
      output_shape.AddDim(input_shape.dim_size(1));
      output_shape.AddDim(input_shape.dim_size(2));
      output_shape.AddDim(input_shape.dim_size(3));
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
    darknet_predict(shadow_pos, input_size,input_flat, output_size, output_flat);
    return;
  }

 private:
  int units_;
  string position_;
};

#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("TeeShadow").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      TeeShadowOp<type>)

//TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);
REGISTER_KERNEL(float);
#undef REGISTER_KERNEL
