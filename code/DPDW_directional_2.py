"""
Created on Tue Aug 16 2021
@author: Allan Perez, School of Physics and Astronomy, University of Glasgow,
2480326p@student.gla.ac.uk

This program aims to investigate the DPDW in a different manner from Linquidst. The inital
idea is to take np.grad accross space at each time stamp, and then take the normal to this
gradient as the "incidence angle".

Mode of use: create directories "vectorfield" and "intensities" in img/gradient/, when
executed, create and move both directories to a directory img/gradient/mask_... where ...
must be replaced with the name of the mask used. E.g. If we used mask x, then we move
vectorfield and intensities directories into img/gradient/mask_x. To execute again, just
repeat the process of creating directories vectorfield and intensities.
"""
from diffuse_imaging_FT import DiffusionSim, BeamGenerator
import numpy as np, matplotlib.pyplot as plt, os
from utils import generate_parameters
from sklearn.feature_selection import mutual_info_regression

def grad_visualize(grid3D, i):
    '''args: 
        grid3D: NxNx2 array, where the last axis is time (2 adjacent time slices).
    returns: None
    '''

    grad1, grad2, grad3 = np.gradient(grid3D)

    plt.figure()
    # The origin argument is required because otherwise the origin (0,0) will be located
    # at the upper left for the imshow, whereas for the vectorfield will be at the lower
    # left.
    plt.imshow(grid3D[:,:,0], origin='lower')
    plt.colorbar()
    plt.title(f"Instantaneous measured intensities (in frame {i})")
    plt.xlabel('X-coordinates (bins)')
    plt.ylabel('Y-coordinates (bins)')
    plt.grid()
    plt.savefig(f'img/gradient/intensities/{i}.png')

    d= np.linspace(0,33,33)
    X,Y = np.meshgrid(d,d)
    plt.figure()
    plt.quiver(X,Y, grad2[:,:,0],grad1[:,:,0], grad3[:,:,0], cmap = 'cool')#'hsv')
    plt.xlabel('X-coordinates (bins)')
    plt.ylabel('Y-coordinates (bins)')
    plt.colorbar()
    plt.title("Gradient of measurement (colors represent magnitude of time gradient)")
    plt.grid()

    plt.savefig(f'img/gradient/vectorfield/{i}.png')
    #plt.show()


if __name__=='__main__':
    # Create parameters
    params = generate_parameters(timeResolution=10e-12)

    # Simulators
    diffSim = DiffusionSim(**params)
    bGen = BeamGenerator(**params)

    # Beam shape
    # Beam's note: It is supposed to be equidistant from the two sources. 
    # So in any case, the detector should be at [0,0]
    input_center = [0,0]        # beam center position (cm)
    input_width = 0.5           # beam width (cm, s.d.)
    beamInput = bGen.intensity_distribution(input_center,
                                                 input_width, 9)
    #bGen.visualize(beamInput2)

    # Mask loading - absorber
    maskp = np.genfromtxt('test_masks/x.txt')
    maskp = np.divide(maskp,255)
    mask = np.ones([diffSim.FoVNumBins, diffSim.FoVNumBins])#mask.shape)
    mask[1:,1:] = maskp
    mask = np.pad(mask,(params["padSize"],)*2,'constant',constant_values=1)

    # Perform simulation
    pl1 = params["propagationLen1"]
    pl2 = params["propagationLen2"]
    outInterface, middleInterface = diffSim.forward_model(beamInput,mask,pl1,pl2)

    l = outInterface.shape[-1]
    if True:
        for i in range(15,l):
            grad_visualize(outInterface[:,:,i:i+2], i)
            print(f'Iteration {i}/{l}')
