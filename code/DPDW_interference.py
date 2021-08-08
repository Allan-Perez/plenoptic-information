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
import numpy as np, matplotlib.pyplot as plt, os

def bin_shift(nu_f, timeResolution):
    timeShift = 1/(2*nu_f)
    return np.int((1/timeResolution)*timeShift)


def process_measurement(outInterface, nu_f, timeRes):
    # Compute Linquidst suggestion of 2D pattern of detectors by combining two conjugate
    # cells' signals by delaying the conjugated cell's signal. To start with it, we use
    # only one conjugate pair: equivalent to the original experiment of 2 sources.
    binShift = bin_shift(nu_f, timeRes)
    detector1 = outInterface[11,16,:]
    detector2 = np.roll(outInterface[21,16,:], binShift)
    signal = detector1+detector2
    signalPadded = np.pad(signal, 200)
    #plt.figure()
    #plt.plot(signalPadded)
    #plt.show()
    return signalPadded

def generate_parameters():
    n = 1.44                   # refractive index of diffuse medium
    xypadding = 65 # Pixel count after 0 padding for the FFTs - avoids wrapping
    params={
        "c" : 3e10/n,             # phase velocity in medium (cm/s)
        "propagationLen1" : 2.5,  # thickness of first diffuse medium (cm)
        "propagationLen2" : 2.5,  # thickness of second diffuse medium (cm)
        "mu_a" : 0.09,     # absorption coefficient (cm^-1)
        "mu_s" : 16.5,     # scattering coefficient (cm^-1)
        "FoV_length" : 4.4,           # size of the camera field of view (cm)
        "timeResolution" : 55e-12,    # camera temporal bin width (ps)
        "timeNumBins" : 251,          # number of temporal bins
        "fieldOfViewBins" : 33,       # number of camera pixels
    }
    params["D"] = (3*(params["mu_a"]+params["mu_s"]))**(-1) # "D" as defined in paper

    # array of positionsin FoV (cm)
    _ = np.linspace(-params["FoV_length"]/2,params["FoV_length"]/2,params["fieldOfViewBins"])
    params["xCoordSpace"] = _
    params["yCoordSpace"] = _
    params["padSize"]= (xypadding - params["fieldOfViewBins"])//2
    return params

def get_nearest_freq_el(nu_f, frequencies):
    # DFT gives bins (sampling), so we need to find the nearest bin
    #print(f"Nu_f {nu_f:.3E} freq max {np.max(frequencies):.3E} freq min {np.min(frequencies):.3E}")
    #plt.figure()
    #plt.plot(frequencies[frequencies>0], 'o')
    #plt.plot(0,nu_f, 'ro')
    #plt.show()
    return np.argmin(np.abs(frequencies-nu_f))

def iterative_roll_measurements(diffSim, beam, iniMask, nu_f, objWidth, pLen1,pLen2,
                                sourceSeparation):
    nIterations = diffSim.FoVNumBins - objWidth
    mask = iniMask#.copy()
    amplitudes=[]
    phases=[]
    cont=True
    detector = (16,16)
    timeDelay = 1/(2*nu_f)

    #directory = "img/"+f"S_{sourceSeparation}_{nu_f}_{objWidth}/" # [S/D]_[distance]_[nu_f]_[objWidth]
    #if not os.path.exists(directory):
    #    os.makedirs(directory)

    for i in range(nIterations):
        print(f"Iteration {i}/{nIterations}")
        outInterface, middleInterface = diffSim.forward_model(beam,mask,pLen1,pLen2)

        print(f"Out interface shape {outInterface.shape}")
        mask = np.roll(mask,1)
        #signal = np.pad(outInterface, ((0,),(0,),(200,)))
        signal = process_measurement(outInterface, nu_f, diffSim.timeRes)
        ftAmplitudes = np.fft.fftn(signal, axes=[-1])
        #ftAmplitudeMidPixel = np.fft.fft(signal)

        #ftAmplitudeMidPixel = np.divide(ftAmplitudeMidPixel,
        #                                np.amax(ftAmplitudeMidPixel))   # normalise
        ftFreq = np.fft.fftfreq(ftAmplitudes.shape[-1], diffSim.timeRes)
        ftFreq = np.fft.fftshift(ftFreq)
        ftAmplitudes = np.fft.fftshift(ftAmplitudes, axes=[-1])
        nearestFreq = get_nearest_freq_el(nu_f, ftFreq)
        ftAmplitudes = ftAmplitudes[nearestFreq]
        ftPhases = np.angle(ftAmplitudes)
        ftAmplitudes = np.abs(ftAmplitudes)
        print(f"Nearest freq {nearestFreq}")

        if False and i%9==0:
            X_, Y_ = np.meshgrid(diffSim.xCoordSpace, diffSim.yCoordSpace)
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(X_,Y_, ftAmplitudes, linewidth=0)
            ax.set_xlabel("X (cm)")
            ax.set_ylabel("Y (cm)")
            ax.set_zlabel("Amplitude of each pixel")
            plt.title(f"Amplitude of nu_f in Freq Space per pixel [nu_f={nu_f:.3E}, \
                     iteration={i+1}/{nIterations}]")
            #plt.savefig(directory+f"amplitude_{i+1}.png")

            X_, Y_ = np.meshgrid(diffSim.xCoordSpace, diffSim.yCoordSpace)
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(X_,Y_, ftPhases, linewidth=0)
            ax.set_xlabel("X (cm)")
            ax.set_ylabel("Y (cm)")
            ax.set_zlabel("Phase of each pixel")
            plt.title(f"Phase of nu_f in Freq Space per pixel [nu_f={nu_f:.3E},\
                     iteration={i+1}/{nIterations}]")
            #plt.savefig(directory+f"phase_{i+1}.png")

            plt.figure()
            plt.imshow(np.sum(middleInterface,-1),interpolation='none')
            plt.show()
            #plt.savefig(directory+f"absorber_{i+1}.png")

        amplitudes.append(ftAmplitudes)
        phases.append(ftPhases)
        #amplitudes.append(np.abs(ftAmplitudeMidPixel[nearestFreq]))
        #phases.append(np.angle(ftAmplitudeMidPixel[nearestFreq]))
    amplitudes, phases = (np.array(amplitudes),np.array(phases))
    print(f"Shapes of time-bins amps {amplitudes.shape}")
    return amplitudes, phases

if __name__=='__main__':
    # Create parameters
    params = generate_parameters()
    ## In the paper they use a detector exactly half-way between sources.
    modulationFrequency = 0.2e9 #Hz - Can it be arbitrary??
    #modulationFrequency = 8e7 # First freq paper
    #modulationFrequency = 6.5e8 # Second freq paper

    # Simulators
    diffSim = DiffusionSim(**params)
    bGen = BeamGenerator(**params)

    # Beam shape
    # Beam's note: It is supposed to be equidistant from the two sources. 
    # So in any case, the detector should be at [0,0]
    input_center = [0,0]        # beam center position (cm)
    input_width = 0.5           # beam width (cm, s.d.)
    bin_delay = bin_shift(modulationFrequency, diffSim.timeRes) #np.int((1/params["timeResolution"])*timeShift)
    d_from_center =  0 #1.5
    beamInput = bGen.intensity_distribution([-d_from_center,0],
                                                 input_width, 9)
    print(f"Position of max: {np.unravel_index(np.argmax(beamInput), beamInput.shape)}")
    print(f"Beam padding: {beamInput.shape}")
    beamInput2 = bGen.intensity_distribution([d_from_center,0],
                                                 input_width, 9+bin_delay)
    print(f"Position of max: {np.unravel_index(np.argmax(beamInput2), beamInput2.shape)}")
    print(f"Beam padding: {beamInput2.shape}")
    #beamInput += beamInput2 # Beams are 3cm apart from each other.
    #bGen.visualize(beamInput2)

    # Mask loading - absorber
    mask = np.genfromtxt('test_masks/tri.txt')
    mask = np.divide(mask,255)
    mask = np.ones([diffSim.FoVNumBins, diffSim.FoVNumBins])#mask.shape)
    objWidth=4
    mask[:,:objWidth] = np.zeros([33,objWidth])
    mask = np.pad(mask,(params["padSize"],)*2,'constant',constant_values=1)

    # Perform iterative simulation
    pl1 = params["propagationLen1"]
    pl2 = params["propagationLen2"]
    amplitudes, phases = iterative_roll_measurements(diffSim,
                                                     beamInput,mask,
                                                     modulationFrequency,
                                                     objWidth,pl1,pl2, d_from_center*2)
    print(phases.shape)

    plt.subplot(1,2,1)
    plt.plot(amplitudes)
    plt.xlabel('Scatterer position in bins(left to right)')
    plt.ylabel('Amplitude of nu_f')
    plt.subplot(1,2,2)
    plt.plot(phases)
    plt.xlabel('Scatterer position in bins(left to right)')
    plt.ylabel('Phase of nu_f')

    plt.show()
