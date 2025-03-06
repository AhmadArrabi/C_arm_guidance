#! python3
"""Minimal projection example with DeepDRR."""

import deepdrr
from deepdrr import geo
from deepdrr.utils import test_utils, image_utils
from deepdrr.projector import Projector
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Circle

class Generate:
    def __init__(self, file, path = "projector.png"):
        """
        Generate class

        Descriptions
        --------------------------------
        generate simulated x-ray image using Deep DRR

        Args
        --------------------------------
        file: str
            file path
        path: str
            output name
        """

        # set volume
        self.patient = deepdrr.Volume.from_nifti(f"{file}", use_thresholding=True)
        self.patient.facedown()
        self.path = path

    def deepdrr_run(self, x, y, z, a, b):
        """
        deepdrr_run
        
        Descriptions
        --------------------------------
        generate simulated x-ray image using Deep DRR

        Args
        --------------------------------
        x: float
            x coordinate value from center of the volume
        y: float
            y coordinate value from center of the volume
        z: float
            z coordinate value from center of the volume
        a: float
            alpha value in radient
        b: float
            beta value in radient

        """

        # define the simulated C-arm
        lower, top = self.patient.get_bounding_box_in_world()
        carm = deepdrr.MobileCArm(geo.Point3D(((top[0] + lower[0]) / 2,
                                       (top[1] + lower[1]) / 2,
                                       (top[2] + lower[2]) / 2, 1)) + geo.v(float(x) ,-float(y), -float(z)), 
                                alpha=-np.rad2deg(float(a)),
                                beta=-np.rad2deg(float(b)))
        center_point = [(top[0] + lower[0]) / 2,(top[1] + lower[1]) / 2,(top[2] + lower[2]) / 2]

        # project in the AP view
        with Projector(self.patient, carm=carm) as projector:

            image = projector()
        # Equalize the image
        equalized_image = exposure.equalize_adapthist(image / np.max(image))

        # Get the center coordinates of the image
        center_x = equalized_image.shape[1] // 2
        center_y = equalized_image.shape[0] // 2

        # Define the radius of the circle
        radius = 20

        # Create a mesh grid for the image coordinates
        y, x = np.ogrid[:equalized_image.shape[0], :equalized_image.shape[1]]

        # Create a mask for the circle
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2

        # Assuming a grayscale image, convert it to RGB by stacking it three times
        rgb_image = np.dstack([equalized_image] * 3)

        # Set the red channel to max (1.0) for the pixels inside the circle
        rgb_image[mask, 0] = 1.0  # Red channel
        rgb_image[mask, 1] = 0.0  # Green channel
        rgb_image[mask, 2] = 0.0  # Blue channel

        # Save the modified image
        plt.imsave('projector.png', rgb_image)

        return rgb_image


    def deepdrr_regenerate(self, x, y, z, a, b):
        """
        deepdrr_run
        
        Descriptions
        --------------------------------
        generate simulated x-ray image using Deep DRR

        Args
        --------------------------------
        x: float
            x coordinate value from center of the volume
        y: float
            y coordinate value from center of the volume
        z: float
            z coordinate value from center of the volume
        a: float
            alpha value in radient
        b: float
            beta value in radient

        """

        # define the simulated C-arm
        lower, top = self.patient.get_bounding_box_in_world()
        carm = deepdrr.MobileCArm(geo.Point3D(((top[0] + lower[0]) / 2,
                                       (top[1] + lower[1]) / 2,
                                       (top[2] + lower[2]) / 2, 1)) + geo.v(float(x) ,-float(y),0), 
                                alpha=-np.rad2deg(float(a)),
                                beta=-np.rad2deg(float(b)))
        coord = geo.Point3D(((top[0] + lower[0]) / 2,
                            (top[1] + lower[1]) / 2,
                            (top[2] + lower[2]) / 2, 1)) + geo.v(float(x), -float(y), 0) - lower

        # project in the AP view
        with Projector(self.patient, 
                       carm=carm, 
                       attenuate_outside_volume=True, 
                       ) as projector:

            image = projector()
        

        image_utils.save(self.path, image)
        return (coord[0], coord[1], coord[2])

    def empty_file(self):
        """
        empty_file
        
        Descriptions
        --------------------------------
        remove previously generated files; create an image with white background
        """
        image_utils.save(self.path, np.ones((1536, 1536))) 


