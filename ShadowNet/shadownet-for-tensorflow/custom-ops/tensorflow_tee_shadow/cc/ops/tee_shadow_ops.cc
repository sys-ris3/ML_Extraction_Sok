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

REGISTER_OP("TeeShadow")
    .Attr("T: realnumbertype")
    .Attr("units: int = 0")
    .Attr("pos: string = 'conv1'")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle input_shape;
      shape_inference::ShapeHandle output_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));

	  // assuming data format NHWC
	  int channel_dim_index = 3;
      int units;
      string shadow_position;
      std::vector<::tensorflow::shape_inference::DimensionHandle> output_dims;
      TF_RETURN_IF_ERROR(c->GetAttr("units", &units));
      TF_RETURN_IF_ERROR(c->GetAttr("pos", &shadow_position));

      // create units dim
      shape_inference::DimensionHandle units_handle = c->MakeDim(units);
      shape_inference::DimensionHandle one_handle = c->MakeDim(1);

      if(shadow_position.compare("results") == 0) { // results shape(batch, channels)
        output_dims.emplace_back(c->Dim(input_shape, 0));
        output_dims.emplace_back(units_handle);
        c->set_output(0, c->MakeShape(output_dims));
      } else if(shadow_position.compare(0, 2, "pw") == 0 ||
        shadow_position.compare(0, 4, "conv") == 0) { 
        TF_RETURN_IF_ERROR(c->ReplaceDim(input_shape, channel_dim_index, units_handle, &output_shape));
        if (shadow_position.compare(0,8,"pwconv13") == 0) {// avgpool
            TF_RETURN_IF_ERROR(c->ReplaceDim(output_shape, 1, one_handle, &output_shape));
            TF_RETURN_IF_ERROR(c->ReplaceDim(output_shape, 2, one_handle, &output_shape));
        }
        c->set_output(0, output_shape);
      } else {
        c->set_output(0, c->input(0));
      }

      return Status::OK();
    })
    .Doc(R"doc(
Forward following non-linear or secret layers to tee ShadowNet for computation.

Attribute "pos" refer to the position of this operation, it affect which tee shadow
model to load. When "pos" is "results", it means the following layer do the prediction. 

Attribute "units" refer to the channel numbers after computation. If following shadow
layers contains LinearTransform Op, then the output channel changes, otherwise the input
shape should be the same as output shape. 

The config for mobilenet. Key(e.g. "conv1") refers to position of the tee shadow layers.
For "conv1", it means this tee shadow op is inserted after "conv1" in the original mobilenet
model. (Ignore the Value(e.g. ("A", 3)) for now.)

    model_config = {"conv1":("A",3),\
                    "dwconv1":("B",9),\
                    "pwconv1":("C",15),\
                    "dwconv2":("B",23),\
                    "pwconv2":("C",29),\
                    "dwconv3":("B",36),\
                    "pwconv3":("C",42),\
                    "dwconv4":("B",50),\
                    "pwconv4":("C",56),\
                    "dwconv5":("B",63),\
                    "pwconv5":("C",69),\
                    "dwconv6":("B",77),\
                    "pwconv6":("C",83),\
                    "dwconv7":("B",90),\
                    "pwconv7":("C",96),\
                    "dwconv8":("B",103),\
                    "pwconv8":("C",109),\
                    "dwconv9":("B",116),\
                    "pwconv9":("C",122),\
                    "dwconv10":("B",129),\
                    "pwconv10":("C",135),\
                    "dwconv11":("B",142),\
                    "pwconv11":("C",148),\
                    "dwconv12":("B",156),\
                    "pwconv12":("C",162),\
                    "dwconv13":("B",169),\
                    "pwconv13":("P",175),\
                    "results":("R",184)}
)doc");
