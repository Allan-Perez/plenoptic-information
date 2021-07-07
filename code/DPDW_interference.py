"""
Created on Tue Jul 6 15:50 2021
@author: Allan Perez, School of Physics and Astronomy, University of Glasgow,
2480326p@student.gla.ac.uk

This program aims to leverage the diffuse_imaging_FT methods to create a simulation from
Lindquist et al (1996). Our aim is to reconstruct the diffuse photon density wave (DPDW)
interference through turbid media to detect inhomogenities (i.e. reconstruct the paper
with a python simulation).

We will simulate two input beams fired at phase shifts pi, pass through a
pixel-wide perfect absorber that moves in one dimension,
and gather the measured interference at the other side with a one-pixel detector (a
time-series essentially), where we can perform the fourier transform and identify the
modulation frequency's evolution as the absorber moves.
"""
from diffuse_imaging_FT import DiffusionSim, BeamGenerator
import numpy as np, matplotlib.pyplot as plt

def generate_parameters():
    n = 1.44                   # refractive index of diffuse medium
    xypadding = 64 # Pixel count after 0 padding for the FFTs - avoids wrapping
    params={
        "c" : 3e10/n,             # phase velocity in medium (cm/s)
        "propagationLen1" : 2.5,  # thickness of first diffuse medium (cm)
        "propagationLen2" : 2.5,  # thickness of second diffuse medium (cm)
        "mu_a" : 0.09,     # absorption coefficient (cm^-1)
        "mu_s" : 16.5,     # scattering coefficient (cm^-1)
        "FoV_length" : 4.4,           # size of the camera field of view (cm)
        "timeResolution" : 55e-12,    # camera temporal bin width (ps)
        "timeNumBins" : 251,          # number of temporal bins
        "fieldOfViewBins" : 32,       # number of camera pixels
    }
    params["D"] = (3*(params["mu_a"]+params["mu_s"]))**(-1) # "D" as defined in paper

    # array of positionsin FoV (cm)
    _ = np.linspace(-params["FoV_length"]/2,params["FoV_length"]/2,params["fieldOfViewBins"])
    params["xCoordSpace"] = _
    params["yCoordSpace"] = _
    params["padSize"]= int((xypadding - params["fieldOfViewBins"])/2)
    return params

def get_nearest_freq_el(nu_f, frequencies):
    # DFT gives bins (sampling), so we need to find the nearest bin
    return np.argmin(np.abs(frequencies-nu_f))

def iterative_roll_measurements(diffSim, beam, iniMask, nu_f, objWidth, pLen1,pLen2):
    nIterations = int(diffSim.FoVNumBins/objWidth)
    mask = iniMask.copy()
    amplitudes=[]
    phases=[]
    for i in range(nIterations):
        print(f"Iteration {i}/{nIterations}")
        outInterface, middleInterface = diffSim.forward_model(beam,mask,pLen1,pLen2)
        mask = np.roll(mask,1)
        ftAmplitudeMidPixel = np.fft.fft(outInterface[16,16])
        ftFreq = np.fft.fftfreq(len(outInterface[16,16]), nu_f)

        nearestFreq = get_nearest_freq_el(nu_f, ftFreq)
        amplitudes.append(np.abs(ftAmplitudeMidPixel[nearestFreq]))
        phases.append(np.angle(ftAmplitudeMidPixel[nearestFreq]))
    return (np.array(amplitudes),np.array(phases))

if __name__=='__main__':
    # Create parameters
    params = generate_parameters()
    centerPixel = (params["fieldOfViewBins"]//2, params["fieldOfViewBins"]//2)
    ## In the paper they use a detector exactly half-way between sources.
    modulationFrequency = 0.1e9 #Hz - Can it be arbitrary??
    timeShift = 1/(2*modulationFrequency)

    # Simulators
    diffSim = DiffusionSim(**params)
    bGen = BeamGenerator(**params)

    # Beam shape
    # Beam's note: It is supposed to be equidistant from the two sources. 
    # So in any case, the detector should be at [0,0]
    input_center = [0,0]        # beam center position (cm)
    input_width = 0.5           # beam width (cm, s.d.)
    pulse_start_bin = 9         # starting bin for the input pulse
    bin_delay = (1/params["timeResolution"])*timeShift
    print(f"The bin delay should be close to an int: {bin_delay}")
    bin_delay = np.int(bin_delay)
    beamInput = bGen.intensity_distribution([-1.5,0],
                                                 input_width, 9)
    beamInput2 = bGen.intensity_distribution([1.5,0],
                                                 input_width, 9+bin_delay)
    beamInput += beamInput2 # Beams are 3cm apart from each other.
    #bGen.visualize(_1+_2)

    # Mask loading - absorber
    mask = np.genfromtxt('test_masks/tri.txt')
    mask = np.divide(mask,255)
    mask = np.ones(mask.shape)
    objWidth=2
    mask[15:17,0:2] = np.zeros([objWidth,objWidth])
    mask = np.pad(mask,(params["padSize"],)*2,'constant',constant_values=1)

    # Perform iterative simulation
    pl1 = params["propagationLen1"]
    pl2 = params["propagationLen2"]
    amplitudes, phases = iterative_roll_measurements(diffSim,
                                                     beamInput,mask,
                                                     modulationFrequency,
                                                     objWidth,pl1,pl2)
    plt.subplot(1, 2, 1)
    plt.plot(amplitudes)
    plt.title("Position vs Ampltidue (from far left to far right)")
    plt.subplot(1,2,2)
    plt.plot(phases)
    plt.title("Position vs Phase (from far left to far right)")
    plt.show()

    # Perform simulation
    ##outInterface, middleInterface = diffSim.forward_model(beamInput,mask,pl1,pl2)
    ##print(outInterface.shape, outInterface[centerPixel].shape, centerPixel)
    ##print(outInterface[16,:,:].shape)

    """
    X_, Y_ = np.meshgrid(np.arange(32),np.arange(251))
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X_,Y_,outInterface[16,:,:].transpose(), linewidth=0, antialiased=False)
    ax.set_xlabel("X (cm) [Cross-section y=16 bin]")
    ax.set_ylabel("Time (bins)")
    ax.set_zlabel("Beam Intensity")
    plt.show()
    """

    """
    nim=3
    plt.subplot(1, nim, 1)
    plt.plot(outInterface[16,16], label="Middle pixel")
    plt.plot(outInterface[30,16], label="Extreme right piexel")
    plt.plot(outInterface[0,16], label="Extreme left pixel")
    plt.title("Detected intensity in end surface, middle pixel")
    plt.xlabel('Time (bins)')
    plt.legend()
    plt.subplot(1, nim, 2)
    plt.imshow(np.sum(middleInterface,2),interpolation='none')
    plt.title("Middle Surface")
    plt.subplot(1, nim, 3)
    #plt.imshow(np.sum(outInterface,2),interpolation='none')
    plt.plot(np.sum(beamInput,(0,1)))
    plt.title("Space-integrated input beam intensity")
    plt.show()
    """
