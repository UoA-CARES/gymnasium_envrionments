import logging
from collections import deque
from functools import cached_property
import matplotlib.pyplot as plt
import cv2
import numpy as np
from envrionments.gym_environment import GymEnvironment

from util.tests.draw import abstract_frame_with_templates
from util.tests.detect import draw_contour


class ImageWrapper:
    def __init__(self, gym: GymEnvironment, k=3):
        self.gym = gym

        self.k = k  # number of frames to be stacked
        self.frames_stacked = deque([], maxlen=k)

        self.frame_width = 224
        self.frame_height = 224
        logging.info("Image Observation is on")

    @cached_property
    def observation_space(self):
        return (9, self.frame_width, self.frame_height)

    @cached_property
    def action_num(self):
        return self.gym.action_num

    @cached_property
    def min_action_value(self):
        return self.gym.min_action_value

    @cached_property
    def max_action_value(self):
        return self.gym.max_action_value

    def set_seed(self, seed):
        self.gym.set_seed(seed)
        
    #apply edge detection
    def apply_edge_detection(self, frame):
            # Split the frame into its component channels
        b, g, r = cv2.split(frame)
        
        # Apply edge detection to each channel
        edges_b = cv2.Canny(b, 200, 600)
        edges_g = cv2.Canny(g, 200, 600)
        edges_r = cv2.Canny(r, 200, 600)
        
        # Stack the channels back together
        edges_rgb = cv2.merge([edges_b, edges_g, edges_r])
    
        
        return edges_rgb

    def grab_frame(self, height=240, width=240):
        frame = self.gym.grab_frame(height=height, width=width)
       
        # frame = self.apply_edge_detection(frame)
        # frame = abstract_frame_with_templates(frame)
        frame = draw_contour(frame)
        
        # actual frame size is 84, 84
        # frame = cv2.resize(frame, (84, 84))
        # cv2.imshow("Current Frame", frame)
        # cv2.waitKey(1)
        return frame
    

    def reset(self):
        _ = self.gym.reset()
        frame = self.grab_frame(height=self.frame_height, width=self.frame_width)
        frame = np.moveaxis(frame, -1, 0)
        for _ in range(self.k):
            self.frames_stacked.append(frame)
        stacked_frames = np.concatenate(list(self.frames_stacked), axis=0)
        return stacked_frames

    def step(self, action):
        _, reward, done, truncated = self.gym.step(action)
        frame = self.grab_frame(height=self.frame_height, width=self.frame_width)
        
        frame = np.moveaxis(frame, -1, 0)
        self.frames_stacked.append(frame)
        stacked_frames = np.concatenate(list(self.frames_stacked), axis=0)
        # uncomment to visualize the current frame
        self.display_current_frame()
        return stacked_frames, reward, done, truncated
    
    def display_current_frame(self):
        # Assuming the latest frame is at the end of the deque and is a single channel due to edge detection
        # and has been reshaped to (3, height, width) by previous operations
        latest_frame = self.frames_stacked[-1]
        
        # Reshape the frame to (height, width, channels)
        latest_frame = np.moveaxis(latest_frame, 0, -1)
        
        # Show frame using cv
        cv2.imshow("Current Frame", latest_frame)
        cv2.waitKey(1)

