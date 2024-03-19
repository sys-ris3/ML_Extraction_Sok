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

REGISTER_OP("TeeShadowGeneric")
    .Attr("T: realnumbertype")
    .Attr("h: int = 0")
    .Attr("w: int = 0")
    .Attr("c: int = 0")
    .Attr("pos: string = 'conv1'")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle input_shape;
      shape_inference::ShapeHandle output_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));

	  // assuming data format NHWC
      int h, w, chn;
	  int channel_dim_index = 3;
      string shadow_position;
      std::vector<::tensorflow::shape_inference::DimensionHandle> output_dims;

      TF_RETURN_IF_ERROR(c->GetAttr("h", &h));
      TF_RETURN_IF_ERROR(c->GetAttr("w", &w));
      TF_RETURN_IF_ERROR(c->GetAttr("c", &chn));
      TF_RETURN_IF_ERROR(c->GetAttr("pos", &shadow_position));

      // create dim
      shape_inference::DimensionHandle h_handle = c->MakeDim(h);
      shape_inference::DimensionHandle w_handle = c->MakeDim(w);
      shape_inference::DimensionHandle chn_handle = c->MakeDim(chn);

      // TODO: Assume the last layer is always named "results"
      // We can also use h=w=0 as indicator for the last tee_shadow layer.
      if(shadow_position.compare("results") == 0) { // results shape(batch, channels)
        output_dims.emplace_back(c->Dim(input_shape, 0));
        output_dims.emplace_back(chn_handle);
        c->set_output(0, c->MakeShape(output_dims));
      } else { 
        TF_RETURN_IF_ERROR(c->ReplaceDim(input_shape, channel_dim_index, chn_handle, &output_shape));
        TF_RETURN_IF_ERROR(c->ReplaceDim(output_shape, 1, h_handle, &output_shape));
        TF_RETURN_IF_ERROR(c->ReplaceDim(output_shape, 2, w_handle, &output_shape));
        c->set_output(0, output_shape);
      }

      return Status::OK();
    })
    .Doc(R"doc(
Forward following non-linear or secret layers to tee ShadowNet for computation.

Attribute "pos" refer to the position of this operation, it affect which tee shadow model to load. When "pos" is "results", it means the following layer do the prediction. 

Attribute "h","w","c" refer to the height, width and  channel numbers after computation respectively.  

If following shadow layers contains LinearTransform Op, then the output channel changes, otherwise the input shape should be the same as output shape. 
)doc");
