
"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
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
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from math import sqrt 

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

# index of person in the model output labeling system
# To detect something else, see for example https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
PERSON_IDX = 1

# Maximum percentage of move between two detections to consider it the same person
MAX_MOVE_PERCENTAGE = 0.2

# Maximum number of frames between two detections to consider it the same person
MAX_NB_FRAMES = 200

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml or pb file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str,
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### Connect to the MQTT client ###
    client = mqtt.Client()
    #client.on_connect = on_connect
    #client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def create_output_image(image, output, width, height, color, confidence, nb_frames):
    '''
    Using the model type, input image, and processed output,
    creates an output image showing the result of inference.
    Also returns the number of detected persons.
    '''
    #The net outputs a blob with the shape: [1, 1, N, 7], where N is the number of detected
    # bounding boxes. For each detection, the description has the format: 
    # [image_id, label, conf, x_min, y_min, x_max, y_max]
    #image_id - ID of the image in the batch
    #label - predicted class ID
    #conf - confidence for the predicted class
    #(x_min, y_min) - coordinates of the top left bounding box corner
    #(x_max, y_max) - coordinates of the bottom right bounding box corner.
    thickness = 1 # in pixels
    detected_persons = []
    #print(output.shape)
    for bounding_box in output[0][0]:
        #print('bounding_box=',bounding_box)
        conf = bounding_box[2]
        # if a person is detected
        if conf >= confidence and PERSON_IDX == int(bounding_box[1]):
            x_min = int(bounding_box[3] * width)
            y_min = int(bounding_box[4] * height)
            x_max = int(bounding_box[5] * width)
            y_max = int(bounding_box[6] * height)
            detected_persons.append([x_min, y_min, x_max, y_max, nb_frames, nb_frames])
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    return image, detected_persons

def is_same_person(p1, p2, max_percent):
    """
    Check if 2 persons (represented by their bounding box coordinates) can be considered as the same person.
    :param p1: first person
    :param p2: second person
    :return: True if they're the same person
    """
    #print('Comparing ', p1, " and ", p2)
    is_same = False
    # if there's not too much time this person have been detected
    if p1[5] - p2[5] < MAX_NB_FRAMES:
        # compute the coordinates of the centers of the bounding boxes
        center_x1 = (p1[0] + p1[2]) / 2
        center_y1 = (p1[1] + p1[3]) / 2
        center_x2 = (p2[0] + p2[2]) / 2
        center_y2 = (p2[1] + p2[3]) / 2
        # compute the distance between the two centers
        distance = sqrt((center_x1 - center_x2)**2 + (center_y1 - center_y2)**2)
        #print('distance=',distance)
        # for comparison, compute the distance of the diagonal of one bounding box
        diag = sqrt((p1[2] - p1[0])**2 + (p1[3] - p1[1])**2)
        # return true if the move is not greater than MAX_MOVE_PERCENTAGE percents of the bounding box diagonal
        #print('(distance / diag)=',(distance / diag))
        if (distance / diag) > max_percent[0] and (distance / diag) < MAX_MOVE_PERCENTAGE:
            max_percent[0] = (distance / diag)
        is_same = (distance / diag) < MAX_MOVE_PERCENTAGE
    return is_same

def get_avg_duration(persons, fps):
    """
    Compute the average duration of detection for an array of persons
    :param persons: array of the detected persons
    :param fps: frame-per-second rate for the video
    :return: None
    """
    if len(persons) > 0:
        total_nb_frames = 0
        for person in persons:
            total_nb_frames = total_nb_frames + person[5] - person[4]   
        # return the average number of frames by person, divided by the FPS rate to get a value in seconds    
        return (total_nb_frames / len(persons)) / fps    
    else:
        return 0

def preprocess_image(image, shape):
    '''
    Pre-process the image as needed
    ''' 
    p_frame = cv2.resize(image, shape)
    p_frame = p_frame.transpose((2,0,1))
    p_frame = p_frame.reshape(1, *p_frame.shape)
    return p_frame
 
def publish_last_duration(previously_detected_persons, client, fps):
    '''
    Publish the duration for the last detected person
    '''
    # previous duration is last person "out" minus "in" frame, divided by fps rate
    if len(previously_detected_persons) > 0:
        previous_duration = (previously_detected_persons[-1][5] - previously_detected_persons[-1][4]) / fps
        client.publish("person/duration", json.JSONEncoder().encode({"duration": previous_duration}))

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # check if we provided a TF model or an IR
    is_tensorflow = os.path.splitext(args.model)[1] == '.pb'
    
    # Initialise the class
    if is_tensorflow:
        from inference_tf import NetworkTf
        infer_network = NetworkTf()
    else:
        from inference import Network
        infer_network = Network()
        
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    
    if not is_tensorflow:
        net_input_shape = infer_network.get_input_shape()

    ### Handle the input ###
    is_single_image_mode = os.path.splitext(args.input)[1] in ['.jpg','.png']

    if not is_single_image_mode:
        cap = cv2.VideoCapture(args.input)
        cap.open(args.input)
        # Grab the shape and FPS rate of the input 
        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)

    # init the total of detected persons
    total = 0
    
    # init the number of frames
    nb_frames = 0
    
    # init the total inference time, to be divided by the number of frames at the end
    total_inference_time = 0
    
    # an array to keep track of previously detected persons
    previously_detected_persons = []
    
    # the average duration of a single person presence
    duration = 0
    
    # fixme  : for debugging
    max_percent = [0]
    
    ### Loop until stream is over ###
    while is_single_image_mode or cap.isOpened():

        if is_single_image_mode:
            print("Single image mode. Analyze ", args.input)
            frame = cv2.imread(args.input)
            height = frame.shape[0]
            width = frame.shape[1]
        else:
            ### Read from the video capture ###
            flag, frame = cap.read()
            nb_frames = nb_frames + 1
            if not flag:
                break
            key_pressed = cv2.waitKey(60)

        if is_tensorflow:
            p_frame = frame
        else:    
            p_frame = preprocess_image(frame, (net_input_shape[3], net_input_shape[2]))
        
        ### Start asynchronous inference for specified request ###
        inference_start = time.time()
        infer_network.exec_net(p_frame)

        ### Wait for the result ###
        if infer_network.wait() == 0:
            
            # record the inference time
            total_inference_time = total_inference_time + (time.time() - inference_start)

            ### Get the results of the inference request ###
            result = infer_network.get_output()
            
            ### Create output frame
            out_frame, detected_persons = create_output_image(frame, result, width, height, (0, 0, 255), float(args.prob_threshold), nb_frames)
            
            if not is_single_image_mode:
                # if there's detected persons in the frame
                count = 0
                if len(detected_persons) > 0:
                    # for each new detection
                    for person in detected_persons:
                        # check if there was a person with a matching bounding box
                        is_new_person = True
                        for index, previous_person in enumerate(previously_detected_persons):
                            # if same person, updating to last coords
                            if is_same_person(person, previous_person, max_percent):
                                # keep the timestamp of the first detection
                                person[4] = previous_person[4]
                                previously_detected_persons[index] = person
                                is_new_person = False
                                break
                        if is_new_person:
                            total = total + 1
                            publish_last_duration(previously_detected_persons, client, fps)
                            previously_detected_persons.append(person)

                #print('previously_detected_persons=',previously_detected_persons)
                #print('max_percent=',max_percent)

                ### Extract any desired stats from the results ###

                ### Calculate and send relevant information on ###
                ### current_count, total_count and duration to the MQTT server ###
                ### Topic "person": keys of "count" and "total" ###
                ### Topic "person/duration": key of "duration" ###
                duration = get_avg_duration(previously_detected_persons, fps)
                #print('count:', len(detected_persons), " total:", len(previously_detected_persons), " person/duration:", duration)
                client.publish("person", json.JSONEncoder().encode({"count": len(detected_persons), "total": len(previously_detected_persons)}))
            
        ### Write an output image if is in single_image_mode ###
        if is_single_image_mode:
            print("Write output file in 'single_image.png'")
            cv2.imwrite('single_image.png', out_frame)
        else:
            ### Send the frame to the FFMPEG server ###
            sys.stdout.buffer.write(out_frame)
            sys.stdout.flush()

        # Break if single_image_mode or escape key pressed
        if is_single_image_mode or key_pressed == 27:
            break
            
    # publish duration for the last detected person
    if not is_single_image_mode:
        publish_last_duration(previously_detected_persons, client, fps)
    
    # Release the capture and destroy any OpenCV windows
    if not is_single_image_mode:
        cap.release()
        cv2.destroyAllWindows()
    
    # Debug mode : Print end results
    #print("Total count of persons : ", total)
    #print("Average presence time : ", duration)
    #print("Number of frames : ", nb_frames)
    #print("Total inference time : ", total_inference_time)
    #print("Inference time by frame : ", total_inference_time / nb_frames)
    

def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)
    # Disconnect from MQTT
    client.disconnect()


if __name__ == '__main__':
    main()