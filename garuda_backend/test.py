#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from config import finalize_configs, cfg
from modeling.generalized_rcnn_hierachy_inner_03 import ResNetFPNModel, ResNetC4Model
from tensorpack.predict import PredictConfig
from tensorpack.tfutils.sessinit import SmartInit
from tensorpack.tfutils import get_default_sess_config

def main():
    # Build the model   
    # Set the configuration to inference mode
    finalize_configs(is_training=False)

    # Instantiate your model
    MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()

    # Create the prediction configuration
    predcfg = PredictConfig(
        model=MODEL,
        session_init=SmartInit('model-7400'),  # Path to your checkpoint
        input_names=MODEL.get_inference_tensor_names()[0],
        output_names=MODEL.get_inference_tensor_names()[1])

    # Start a session with the default session configuration
    with tf.Session(config=get_default_sess_config()) as sess:
        # Initialize the model variables from the checkpoint
        predcfg.session_init.init(sess)

        # Get the input and output tensors by name
        input_tensor = tf.get_default_graph().get_tensor_by_name(predcfg.input_names[0] + ':0')
        output_tensors = [tf.get_default_graph().get_tensor_by_name(name + ':0') for name in predcfg.output_names]

        # Export as SavedModel
        export_path = '/path/to/export/saved_model'  # Replace with your desired export path
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        # Define inputs and outputs
        inputs = {'input': tf.saved_model.utils.build_tensor_info(input_tensor)}
        outputs = {}
        for name, tensor in zip(predcfg.output_names, output_tensors):
            outputs[name] = tf.saved_model.utils.build_tensor_info(tensor)

        # Build the signature
        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )

        # Add the meta graph and variables
        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
            }
        )
        builder.save()

    print(f"SavedModel exported to: {export_path}")

if __name__ == "__main__":
    main()
