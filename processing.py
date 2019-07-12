

import pickle
import numpy as np
from PIL import Image
from config import INPUT_DIM, ROI
from vae.controller import VAEController



class Vae:

    def __init__(self, ):

        self.n_commands = 2
        self.n_command_history = 20
        self.command_history = np.zeros((1, self.n_commands * self.n_command_history))


    def preprocessing(self, image):


        # Resize and crop image
        image = np.array(image)

        # Region of interest
        r = ROI
        image = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        # Convert RGB to BGR
        image = image[:, :, ::-1]
        image_array = image


        return image_array

    def load_vae(self, path = None, z_size = None):

        """
        :param path: (str)
        :param z_size: (int)
        :return: (VAEController)
        """
        # z_size will be recovered from saved model
        if z_size is None:
            assert path is not None

        vae = VAEController(z_size=z_size)

        if path is not None:
            vae.load(path)

        print("Dim VAE = {}".format(vae.z_size))

        return vae


    def postprocessing(self, action, observation):

        if self.n_command_history > 0:
            
            self.command_history = np.roll(self.command_history, shift=-self.n_commands, axis=-1)
            self.command_history[..., -self.n_commands:] = action

            print(self.command_history)
            observation = np.concatenate((observation, self.command_history), axis=-1)

        return observation, self.command_history





'''
or_image = Image.open("1.jpg")

Vae = Vae()
vae_32 = Vae.load_vae(path = "vae-32")

for i in range(40):

    image = Vae.preprocessing(or_image)
  
    image = vae_32.encode(image)

    print(Vae.postprocessing((i * -0.215453, i * 0.598416), image).shape)

'''


'''
#img_string = data["image"]


#image = Image.open(BytesIO(base64.b64decode(img_string)))
# Resize and crop image
image = np.array(image)
# Save original image for render
original_image = np.copy(image)
# Resize if using higher resolution images
#image = cv2.resize(image, (120, 160))


# Region of interest
r = ROI
image = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
# Convert RGB to BGR
image = image[:, :, ::-1]
image_array = image

print(image_array.shape)

def load_vae(path=None, z_size=None):
    """
    :param path: (str)
    :param z_size: (int)
    :return: (VAEController)
    """
    # z_size will be recovered from saved model
    if z_size is None:
        assert path is not None

    vae = VAEController(z_size=z_size)
    if path is not None:
        vae.load(path)
    print("Dim VAE = {}".format(vae.z_size))
    return vae

vae = load_vae(path = "vae-32")
image = vae.encode(image)

print(image.shape)

n_commands = 2
n_command_history = 20
command_history = np.zeros((1, n_commands * n_command_history))

print(command_history)


def postprocessing(action, observation, command_history):


    command_history = command_history

    if n_command_history > 0:
            
        command_history = np.roll(command_history, shift=-n_commands, axis=-1)
        command_history[..., -n_commands:] = action

        print(command_history)
        observation = np.concatenate((observation, command_history), axis=-1)

    return observation


print(postprocessing((-0.215453, 0.598416), image, command_history).shape)
'''

