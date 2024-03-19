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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("LinearTransform")
    .Attr("T: realnumbertype")
    .Input("input: T")
    .Input("weights: int32")
    .Input("scalar: T")
    .Output("transformed: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle input_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));

      shape_inference::ShapeHandle weight_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &weight_shape));

      shape_inference::ShapeHandle scalar_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &scalar_shape));
                
      // Get number of weight channel
      shape_inference::DimensionHandle weight_channel_dim = c->Dim(weight_shape, 1);

	  // assuming data format NHWC
	  int channel_dim_index = 3;
      shape_inference::ShapeHandle output_shape;
      TF_RETURN_IF_ERROR(c->ReplaceDim(input_shape, channel_dim_index, weight_channel_dim, &output_shape));

      c->set_output(0, output_shape);

      return Status::OK();
    })
    .Doc(R"doc(
Apply linear transform on input tensor with weights and scalar.
Assume weights has two dimension, recording from_idx and rand_idx in
input Tensor. The output will be computed with the following formula:

output[h][w][c] = input[h][w][from_idx]*scalar[c] + input[h][w][rand_idx]. 

transformed: The transformed output Tensor. 
)doc");
