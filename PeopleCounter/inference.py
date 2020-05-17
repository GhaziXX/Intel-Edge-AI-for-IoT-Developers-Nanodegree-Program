#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore, IEPlugin

DEFAULT_CPU_EXTENSION_LINUX = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### Initialize any class variables desired ###
        self.core = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None

    def load_model(self, model, device, cpu_extension=DEFAULT_CPU_EXTENSION_LINUX):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
        ### Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        # Initialize the plugin
        if device:
            plugin = IEPlugin(device)
        else:
            plugin = IEPlugin(device="CPU")
            
        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)
        
        ### Check for supported layers ###
        supported_layers = plugin.get_supported_layers(self.network)
        needed_layers = self.network.layers.keys()
        unsupported_layers = set(needed_layers) - set(supported_layers)
        
        self.core = IECore()
        
        ### Add any necessary extensions ###
        if len(unsupported_layers) > 0:
            if cpu_extension and device and "CPU" in device:
                self.core.add_extension(cpu_extension, device)
            else:
                print("Error : there's unsupported layers in this model. Please use the -d argument and specify a supporting device")
                print('Unsupported_layers: ', unsupported_layers)
                sys.exit(1)

        # Load the IENetwork into the plugin
        self.exec_network = self.core.load_network(self.network, device)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
            
        ### Return the loaded inference plugin ###
        return self.core

    def get_input_shape(self):
        '''
        Gets the input shape of the network
        '''
        ### Return the shape of the input layer ###
        return self.network.inputs[self.input_blob].shape

    def exec_net(self, image):
        ### Start an asynchronous request ###
        self.exec_network.start_async(request_id=0, inputs={self.input_blob: image})
        ### TODO: Return any necessary information ###
        return

    def wait(self):
        '''
        Wait for the request to be complete.
        '''
        #while True:
        #    status = self.exec_network.requests[0].wait(-1)
        #    if status == 0:
        #        break
        #    else:
        #        time.sleep(1)
        status = self.exec_network.requests[0].wait(-1)
        ### Wait for the request to be complete. ###
        ### Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return status

    def get_output(self):
        ### Extract and return the output results
        return self.exec_network.requests[0].outputs[self.output_blob]
        ### Note: You may need to update the function parameters. ###