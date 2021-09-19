"""
Created on Tue Aug 16 2021
@author: Allan Perez, School of Physics and Astronomy, University of Glasgow,
2480326p@student.gla.ac.uk

This program aims to investigate the DPDW in a different manner from Linquidst. The inital
idea is to take np.grad accross space at each time stamp, and then take the normal to this
gradient as the "incidence angle".
"""
from diffuse_imaging_FT import DiffusionSim, BeamGenerator
import numpy as np, matplotlib.pyplot as plt, os
from utils import generate_parameters
from sklearn.feature_selection import mutual_info_regression

def correct_angle(dy,dx,dg):
    theta = np.arcsin(np.abs(dy)/dg)
    if dy>0 and dx<0:
        return np.pi - theta
    if dy<0 and dx<0:
        return np.pi + theta
    if dy<0 and dx>0:
        return 2*np.pi - theta
    return theta

def grad_visualize(grid2D, i):
    grad1, grad2 = np.gradient(grid2D) #Grad 1 is presumably the y-axis
    """
    gradP = np.sqrt(np.square(grad1) + np.square(grad2))
    gradT = np.arcsin(np.divide(grad1, gradP))
    correct_angleV = np.vectorize(correct_angle)
    gradT = correct_angleV(grad1, grad2, gradP)
    #gradT *=  1/gradT.max() #normalize between 0 and 1
    print(f"Check: {gradP.shape}")
    """
    """
    plt.imshow(grad1)
    plt.title('Gradient axis 0')
    plt.savefig(f'img/gradient/axis_0/{i}.png')

    plt.imshow(grad2)
    plt.title('Gradient axis 1')
    plt.savefig(f'img/gradient/axis_1/{i}.png')
    """

    plt.figure()
    # The origin argument is required because otherwise, the origin (0,0) will be located
    # at the upper left for the imshow, whereas for the vectorfield will be at the lower
    # left.
    plt.imshow(grid2D, origin='lower')
    plt.colorbar()
    plt.title(f"Instantaneous measured intensities (in frame {i})")
    plt.xlabel('X-coordinates (bins)')
    plt.ylabel('Y-coordinates (bins)')
    plt.grid()
    #plt.savefig(f'img/gradient/intensities/{i}.png')

    """
    plt.figure()
    plt.imshow(gradP, cmap=plt.cm.BuPu_r)
    plt.title('Actual gradient (sqrt sum of squares of axis)')
    plt.colorbar()
    #plt.savefig(f'img/gradient/axis_2/{i}.png')
    plt.figure()
    plt.imshow(gradT, cmap=plt.cm.BuPu_r)
    plt.title(f"Grad Direction (theta from x-axis) min:{gradT.min():.2}, max:{gradT.max():.2}")
    #plt.savefig(f'img/gradient/axis_3/{i}.png')
    plt.colorbar()
    """

    d= np.linspace(0,33,33)
    X,Y = np.meshgrid(d,d)
    plt.figure()
    plt.quiver(X,Y, grad2,grad1, gradP, cmap = 'cool')#'hsv')
    plt.xlabel('X-coordinates (bins)')
    plt.ylabel('Y-coordinates (bins)')
    plt.colorbar()
    plt.title("Gradient of measurement (colors represent scale)")
    plt.grid()

    #plt.savefig(f'img/gradient/vectorfield/{i}.png')
    plt.show()


if __name__=='__main__':
    # Create parameters
    params = generate_parameters()

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
    """
    Z=outInterface.argmax(axis=-1)

    print(f"First timebin max: {Z.min()}")
    X = np.arange(outInterface.shape[0])
    Y = np.arange(outInterface.shape[1])
    X,Y = np.meshgrid(X, Y)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X,Y,Z, linewidth=0, antialiased=False)
    ax.set_zlabel('Time Bin of max(phi)', rotation = 0)
    ax.set_ylabel('Spatial Y (Column bin)', rotation = 0)
    ax.set_xlabel('Spatial X (Row bin)', rotation = 0)
    plt.show()
    """

    mInterface = middleInterface[15:48,15:48,:]
    xn, yn, tn = mInterface.shape
    features = outInterface.reshape(xn*yn, tn).T#mInterface.reshape(xn*yn,tn).T
    def mi_calc(t):
        print(f"TShape: {t.shape}, FShape: {features.shape}")
        mi = mutual_info_regression(features, t)
        print(f"Mi: {mi}, {mi.sum()}, {mi.shape}")
        miIm = mi.reshape(33,33)
        plt.imshow(miIm)
        plt.colorbar()
        plt.show()
        return mi


    #vmi_calc = np.vectorize(mi_calc)
    mis = np.apply_along_axis(mi_calc, -1, mInterface) #outInterface)
    #mis = vmi_calc(outInterface)

    print(f"MIs: {mis.shape}")
    #plt.plot(mi)
    #plt.show()

    l = outInterface.shape[-1]
    if False:
        for i in range(10,l):
            grad_visualize(outInterface[:,:,i], i)
            print(f'Iteration {i}/{l}')
