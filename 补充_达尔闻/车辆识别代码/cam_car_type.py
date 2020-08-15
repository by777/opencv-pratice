import cv2
import numpy as np
import openvino
# import argparse
import cv2
import numpy as np
import time

# import matplotlib.pyplot as plt
# %matplotlib inline

'''
Contains code for working with the Inference Engine.
You'll learn how to implement this code and more in
the related lesson on the topic.
'''

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore

class Network:
    '''
    Load and store information for working with the Inference Engine,
    and any loaded models.
    '''

    def __init__(self):
        self.plugin = None
        self.input_blob = None
        self.exec_network = None


    def load_model(self, model, device="CPU", cpu_extension=None):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Initialize the plugin
        self.plugin = IECore()

        # Add a CPU extension, if applicable
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)

        # Read the IR as a IENetwork
        # network = IENetwork(model=model_xml, weights=model_bin)
        network = self.plugin.read_network(model=model_xml, weights=model_bin)

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(network, device)

        # Get the input layer
        self.input_blob = next(iter(network.inputs))

        # Return the input shape (to determine preprocessing)
        return network.inputs[self.input_blob].shape


    def sync_inference(self, image):
        '''
        Makes a synchronous inference request, given an input image.
        '''
        self.exec_network.infer({self.input_blob: image})
        return


    def extract_output(self):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        return self.exec_network.requests[0].outputs

# Create a Network for using the Inference Engine
inference_network = Network()

n, c, h, w = inference_network.load_model('models/vehicle-attributes-recognition-barrier-0039.xml', 'CPU', None)

# 神经棒
# n, c, h, w = inference_network.load_model('models/vehicle-attributes-recognition-barrier-0039.xml', 'MYRIAD', None)

# # Read the input image
# image = cv2.imread(image_path)

def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image

CAR_COLORS = ["white", "gray", "yellow", "red", "green", "blue", "black"]
CAR_TYPES = ["car", "bus", "truck", "van"]

def process_frame(frame):
    START_TIME = time.time()
    preprocessed_image = preprocessing(frame, h, w)
    # Perform synchronous inference on the image
    inference_network.sync_inference(preprocessed_image)
    # Obtain the output of the inference request
    output = inference_network.extract_output()
    color = output['color'].flatten()
    car_type = output['type'].flatten()
    color_pred = np.argmax(color)
    type_pred = np.argmax(car_type)
    processed_output = color_pred, type_pred
    # 获取颜色和类型预测结果
    color = CAR_COLORS[processed_output[0]]
    car_type = CAR_TYPES[processed_output[1]]  
    # 获取颜色和类型置信度
    color_score = output['color'].flatten()[processed_output[0]]
    type_score = output['type'].flatten()[processed_output[1]]
    # Scale the output text by the image shape
    scaler = max(int(frame.shape[0] / 1000), 1)
    frame = cv2.putText(frame, 
        "Type: {:^8} Score:{:.3f}".format(car_type, type_score), 
        (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 
        1 * scaler, (255, 255, 255), 2 * scaler)

    frame = cv2.putText(frame, 
        "Color:{:^8} Score:{:.3f}".format(color, color_score), 
        (25 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 
        1 * scaler, (255, 255, 255), 2 * scaler)
    FPS = 1/(time.time() - START_TIME)
    frame = cv2.putText(frame, 
        "FPS:{:.3f}".format(FPS), 
        (25 * scaler, 150 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 
        1 * scaler, (255, 255, 255), 2 * scaler)
    return frame

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.open(0)

while cap.isOpened():
    flag, frame = cap.read()
    if not flag:
        break
    key_pressed = cv2.waitKey(60)
    
    ### 对摄像头捕获的画面进行处理
    frame = process_frame(frame)
    ### 处理完毕
    cv2.imshow('car_type_recognition',frame)
    # Break if escape key pressed
    if key_pressed == 27:
        break
    
cap.release()
cv2.destroyAllWindows()