# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

The process behind converting custom layers involves using the tools provided with the model opimizer to extract the wanted layer.then implementing the layer (Using GPUs, CPUs..) and then run the conversion while specifiying layers implementation.

Some of the potential reasons for handling custom layers are:
  - Implement a layer in a diffrent way than the provided by the optimizer.
  - Use a layer in the source layer which isn't supported by the optimizer.

## Comparing Model Performance

I used the ### ssd_mobilenet_v2_coco ### because it's the basic model to use, downloaded from: 
http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
and i converted it using the command 
`python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels`
I tried it with a basic version of app.py and it gave me the following results:
  - 736 detected persons on the test video (using 6% probability threshold)
  - 3m16s using the time command
## Assess Model Use Cases

Some of the potential use cases of the people counter app are:
  - knows how many persons comes into a specific place
  - to count people entering a place

Each of these use cases would be useful because...

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows:
  - the loss goes up when the brightness decreases
  - the loss goes up when lowering the resolution, that could be a good thing if lilmited by bandwith and have a limited resources
