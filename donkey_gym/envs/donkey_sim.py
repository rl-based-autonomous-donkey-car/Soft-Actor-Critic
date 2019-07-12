# Original author: Tawn Kramer

import asyncore
import base64
import math
import time
from io import BytesIO
from threading import Thread

import numpy as np
from PIL import Image

from config import INPUT_DIM, ROI, THROTTLE_REWARD_WEIGHT, MAX_THROTTLE, MIN_THROTTLE, \
    REWARD_CRASH, CRASH_SPEED_WEIGHT
from donkey_gym.core.fps import FPSTimer
from donkey_gym.core.tcp_server import IMesgHandler, SimServer


import cv2
import numpy as np  
def nothing(x):
    pass

'''
cv2.namedWindow("Tracking")
cv2.createTrackbar("LHW", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LSW", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LVW", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UHW", "Tracking", 255, 255, nothing)
cv2.createTrackbar("USW", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UVW", "Tracking", 255, 255, nothing)

cv2.createTrackbar("LHY", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LSY", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LVY", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UHY", "Tracking", 255, 255, nothing)
cv2.createTrackbar("USY", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UVY", "Tracking", 255, 255, nothing)
'''
def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension        
    return cv2.bitwise_and(image, mask)

    
def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left  = [cols*.00000001,rows*0.95]
    top_left     = [cols*0.1, rows*0.6]
    bottom_right = [cols*0.9, rows*0.95]
    top_right    = [cols*0.9, rows*0.2] 
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return vertices


# images showing the region of interest only

def select_rgb_white_yellow(image):
    l_h_w = cv2.getTrackbarPos("LHW", "Tracking")
    l_s_w = cv2.getTrackbarPos("LSW", "Tracking")
    l_v_w = cv2.getTrackbarPos("LVW", "Tracking")
   
    u_h_w = cv2.getTrackbarPos("UHW", "Tracking")
    u_s_w = cv2.getTrackbarPos("USW", "Tracking")
    u_v_w = cv2.getTrackbarPos("UVW", "Tracking")


    l_h_y = cv2.getTrackbarPos("LHY", "Tracking")
    l_s_y= cv2.getTrackbarPos("LSY", "Tracking")
    l_v_y= cv2.getTrackbarPos("LVY", "Tracking")
   
    u_h_y= cv2.getTrackbarPos("UHY", "Tracking")
    u_s_y= cv2.getTrackbarPos("USY", "Tracking")
    u_v_y= cv2.getTrackbarPos("UVY", "Tracking")

    
    
    lower1 = np.uint8([0, 0, 0])
    upper1 = np.uint8([255, 255, 255])

    #lower = np.uint8([200, 200, 200])
    #upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower1, upper1)
    # yellow color mask
    lower2 = np.uint8([0, 0, 0])
    upper2 = np.uint8([255, 255, 255])

    #lower = np.uint8([190, 190,   0])
    #upper = np.uint8([255, 255, 255])

    yellow_mask = cv2.inRange(image, lower2, upper2)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    #masked=cv2.GaussianBlur(masked, (15, 15), 0)

    masked=cv2.Canny(masked[:,:], 200, 300)

    return masked


def segment(image):


    frame = cv2.imread(image)
    res=select_rgb_white_yellow(frame)

    return res



class DonkeyUnitySimContoller:
    """
    Wrapper for communicating with unity simulation.

    :param level: (int) Level index
    :param port: (int) Port to use for communicating with the simulator
    :param max_cte_error: (float) Max cross track error before reset
    """

    def __init__(self, level, port=9090, max_cte_error=3.0):
        self.level = level
        self.verbose = False

        # sensor size - height, width, depth
        self.camera_img_size = INPUT_DIM

        self.address = ('0.0.0.0', port)

        # Socket message handler
        self.handler = DonkeyUnitySimHandler(level, max_cte_error=max_cte_error)
        # Create the server to which the unity sim will connect
        self.server = SimServer(self.address, self.handler)
        # Start the Asynchronous socket handler thread
        self.thread = Thread(target=asyncore.loop)
        self.thread.daemon = True
        self.thread.start()

    def close_connection(self):
        return self.server.handle_close()

    def wait_until_loaded(self):
        """
        Wait for a client (Unity simulator).
        """
        while not self.handler.loaded:
            print("Waiting for sim to start..."
                  "if the simulation is running, press EXIT to go back to the menu")
            time.sleep(3.0)


    def reset(self):
        self.handler.reset()

    def get_sensor_size(self):
        """
        :return: (int, int, int)
        """
        return self.handler.get_sensor_size()

    def take_action(self, action):
        self.handler.take_action(action)

    def observe(self):
        """
        :return: (np.ndarray)
        """
        return self.handler.observe()

    def quit(self):
        pass

    def render(self, mode):
        pass

    def is_game_over(self):
        return self.handler.is_game_over()

    def calc_reward(self, done):
        return self.handler.calc_reward(done)


class DonkeyUnitySimHandler(IMesgHandler):
    """
    Socket message handler.

    :param level: (int) Level ID
    :param max_cte_error: (float) Max cross track error before reset
    """

    def __init__(self, level, max_cte_error=3.0):
        self.level_idx = level
        self.sock = None
        self.loaded = False
        self.verbose = False
        self.timer = FPSTimer(verbose=0)
        self.max_cte_error = max_cte_error

        # sensor size - height, width, depth
        self.camera_img_size = INPUT_DIM
        self.image_array = np.zeros(self.camera_img_size)
        self.original_image = None
        self.last_obs = None
        self.last_throttle = 0.0
        # Disabled: hit was used to end episode when bumping into an object
        self.hit = "none"
        # Cross track error
        self.cte = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.steering_angle = 0.0
        self.current_step = 0
        self.speed = 0
        self.steering = None

        # Define which method should be called
        # for each type of message
        self.fns = {'telemetry': self.on_telemetry,
                    "scene_selection_ready": self.on_scene_selection_ready,
                    "scene_names": self.on_recv_scene_names,
                    "car_loaded": self.on_car_loaded}

    def on_connect(self, socket_handler):
        """
        :param socket_handler: (socket object)
        """
        self.sock = socket_handler

    def on_disconnect(self):
        """
        Close socket.
        """
        self.sock.close()
        self.sock = None

    def on_recv_message(self, message):
        """
        Distribute the received message to the appropriate function.

        :param message: (dict)
        """
        if 'msg_type' not in message:
            print('Expected msg_type field')
            return

        msg_type = message['msg_type']
        if msg_type in self.fns:
            self.fns[msg_type](message)
        else:
            print('Unknown message type', msg_type)

    def reset(self):
        """
        Global reset, notably it
        resets car to initial position.
        """
        if self.verbose:
            print("resetting")
        self.image_array = np.zeros(self.camera_img_size)
        self.last_obs = None
        self.hit = "none"
        self.cte = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.current_step = 0
        self.send_reset_car()
        self.send_control(0, 0)
        time.sleep(1.0)
        self.timer.reset()

    def get_sensor_size(self):
        """
        :return: (tuple)
        """
        return self.camera_img_size

    def take_action(self, action):
        """
        :param action: ([float]) Steering and throttle
        """
        if self.verbose:
            print("take_action")

        throttle = action[1]
        self.steering = action[0]
        self.last_throttle = throttle
        self.current_step += 1

        self.send_control(self.steering, throttle)

    def observe(self):
        while self.last_obs is self.image_array:
            time.sleep(1.0 / 120.0)

        self.last_obs = self.image_array
        observation = self.image_array
        done = self.is_game_over()
        reward = self.calc_reward(done)
        info = {}

        self.timer.on_frame()

        return observation, reward, done, info


    def is_game_over(self):
        """
        :return: (bool)
        """
        return self.hit != "none" or math.fabs(self.cte) > self.max_cte_error

    def calc_reward(self, done):
        """
        Compute reward:
        - +1 life bonus for each step + throttle bonus
        - -10 crash penalty - penalty for large throttle during a crash

        :param done: (bool)
        :return: (float)
        """
        if done:
            # penalize the agent for getting off the road fast
            norm_throttle = (self.last_throttle - MIN_THROTTLE) / (MAX_THROTTLE - MIN_THROTTLE)
            return REWARD_CRASH - CRASH_SPEED_WEIGHT * norm_throttle
        # 1 per timesteps + throttle
        throttle_reward = THROTTLE_REWARD_WEIGHT * (self.last_throttle / MAX_THROTTLE)
        return 1 + throttle_reward

    # ------ Socket interface ----------- #

    def on_telemetry(self, data):
        """
        Update car info when receiving telemetry message.

        :param data: (dict)
        """
        img_string = data["image"]



        image = Image.open(BytesIO(base64.b64decode(img_string)))


        # Resize and crop image
        image = np.array(image)



        image = segment(image)
        #image = segment(image)
        # Save original image for render
        self.original_image = np.copy(image)
        # Resize if using higher resolution images
        #image = cv2.resize(image, (120, 160))


        # Region of interest
        r = ROI
        image = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        # Convert RGB to BGR
        image = image[:, :, ::-1]
        self.image_array = image


        #self.image_array = segment(self.image_array)


        # Here resize is not useful for now (the image have already the right dimension)
        #self.image_array = cv2.resize(image, (120, 160))

        # name of object we just hit. "none" if nothing.
        # NOTE: obstacle detection disabled
        # if self.hit == "none":
        #     self.hit = data["hit"]

        self.x = data["pos_x"]
        self.y = data["pos_y"]
        self.z = data["pos_z"]
        self.steering_angle = data['steering_angle']
        self.speed = data["speed"]

        # Cross track error not always present.
        # Will be missing if path is not setup in the given scene.
        # It should be setup in the 3 scenes available now.
        try:
            self.cte = data["cte"]
            # print(self.cte)
        except KeyError:
            print("No Cross Track Error in telemetry")
            pass

    def on_scene_selection_ready(self, _data):
        """
        Get the level names when the scene selection screen is ready
        """
        print("Scene Selection Ready")
        self.send_get_scene_names()

    def on_car_loaded(self, _data):
        if self.verbose:
            print("Car Loaded")
        self.loaded = True

    def on_recv_scene_names(self, data):
        """
        Select the level.

        :param data: (dict)
        """
        if data is not None:
            names = data['scene_names']
            if self.verbose:
                print("SceneNames:", names)
            self.send_load_scene(names[self.level_idx])

    def send_control(self, steer, throttle):
        """
        Send message to the server for controlling the car.

        :param steer: (float)
        :param throttle: (float)
        """
        if not self.loaded:
            return
        msg = {'msg_type': 'control', 'steering': steer.__str__(), 'throttle': throttle.__str__(), 'brake': '0.0'}
        self.queue_message(msg)

    def send_reset_car(self):
        """
        Reset car to initial position.
        """
        msg = {'msg_type': 'reset_car'}
        self.queue_message(msg)

    def send_get_scene_names(self):
        """
        Get the different levels availables
        """
        msg = {'msg_type': 'get_scene_names'}
        self.queue_message(msg)

    def send_load_scene(self, scene_name):
        """
        Load a level.

        :param scene_name: (str)
        """
        msg = {'msg_type': 'load_scene', 'scene_name': scene_name}
        self.queue_message(msg)

    def send_exit_scene(self):
        """
        Go back to scene selection.
        """
        msg = {'msg_type': 'exit_scene'}
        self.queue_message(msg)

    def queue_message(self, msg):
        """
        Add message to socket queue.

        :param msg: (dict)
        """
        if self.sock is None:
            if self.verbose:
                print('skipping:', msg)
            return

        if self.verbose:
            print('sending', msg)
        self.sock.queue_message(msg)
