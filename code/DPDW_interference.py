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

def iterative_roll_measurements(diffSim, beam, iniMask, nu_f, objWidth, pLen1,pLen2):
    nIterations = diffSim.FoVNumBins - objWidth
    mask = iniMask#.copy()
    amplitudes=[]
    phases=[]
    cont=True
    detector = (16,16)
    # Perhaps fft is being done not in t axis
    for i in range(nIterations):
        print(f"Iteration {i}/{nIterations}")
        outInterface, middleInterface = diffSim.forward_model(beam,mask,pLen1,pLen2)

        print(f"Out interface shape {outInterface.shape}")
        mask = np.roll(mask,1)
        #ftAmplitudeMidPixel = np.fft.fftn(outInterface, axes=[-1])
        signal = np.pad(outInterface[detector], 10)
        ftAmplitudeMidPixel = np.fft.fft(signal)

        #ftAmplitudeMidPixel = np.divide(ftAmplitudeMidPixel,
        #                                np.amax(ftAmplitudeMidPixel))   # normalise
        print(f"Out interface fft shape {ftAmplitudeMidPixel .shape}")
        ftFreq = np.fft.fftfreq(signal.shape[-1], diffSim.timeRes)
        print(f"Freq fft shape {ftFreq.shape}")
        ftFreq = np.fft.fftshift(ftFreq)
        ftAmplitudeMidPixel = np.fft.fftshift(ftAmplitudeMidPixel)
        nearestFreq = get_nearest_freq_el(nu_f, ftFreq)
        print(f"Nearest freq {nearestFreq}")

        if i%8==0:
            print(f"Max out interference index: {np.argmax(np.sum(outInterface[16,:,],1))}")
            plt.figure()
            plt.subplot(2,2, 1)
            plt.title(f"Detected dispersion curve (Iter {i}/{nIterations}).")
            plt.plot(outInterface[detector])
            plt.subplot(2,2,2)
            plt.imshow(np.sum(middleInterface,2),interpolation='none')
            plt.subplot(2,2,3)
            plt.plot(np.sum(outInterface[16,:,],1))
            plt.show()
            plt.figure(figsize=(12,6))
            plt.title(f"Frequency spectrum [with detector position {detector}]")
            plt.scatter(ftFreq, np.abs(ftAmplitudeMidPixel), color='b', alpha=0.7)
            plt.scatter(ftFreq[nearestFreq], np.abs(ftAmplitudeMidPixel[nearestFreq]), color='r',
                        label=f'Closest Modulation Frequency [{ftFreq[nearestFreq]}]'
                        )
            plt.scatter(nu_f, np.abs(ftAmplitudeMidPixel[nearestFreq]), color='g')
            plt.legend()
            plt.grid(True)
            plt.xlim(-1e9, 1e9)
            plt.show()


        if False: #cont!='n':
            plt.figure()
            plt.subplot(2,2, 1)
            plt.imshow(np.sum(middleInterface,2),interpolation='none')
            plt.title("Middle Surface")
            plt.subplot(2,2, 2)
            plt.imshow(np.sum(outInterface,2))
            plt.title("Measurement surface")
            plt.subplot(2,2,3)
            plt.plot(ftFreq,np.abs(ftAmplitudeMidPixel))
            plt.plot(nearestFreq, np.abs(ftAmplitudeMidPixel)[nearestFreq], 'ro')
            plt.subplot(2,2,4)
            plt.plot(outInterface[15,15])
            plt.show()
            cont = input("Cont?")
        if False and cont!='n':
            X_, Y_ = np.meshgrid(diffSim.xCoordSpace, diffSim.yCoordSpace)

            times_max = np.argmax(np.abs(outInterface), axis=-1)

            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(X_,Y_, times_max, linewidth=0)
            ax.set_xlabel("X (cm)")
            ax.set_ylabel("Y (cm)")
            ax.set_zlabel("Time of first max (bins)")
            plt.show()

            plt.figure()
            plt.imshow(np.sum(middleInterface,2),interpolation='none')
            plt.title('Mask position')
            plt.show()

        #amplitudes.append(np.abs(ftAmplitudeMidPixel[15,:,nearestFreq]))
        #phases.append(np.angle(ftAmplitudeMidPixel[15,:,nearestFreq]))
        amplitudes.append(np.abs(ftAmplitudeMidPixel[nearestFreq]))
        phases.append(np.angle(ftAmplitudeMidPixel[nearestFreq]))
    amplitudes, phases = (np.array(amplitudes),np.array(phases))
    print(f"Shapes of time-bins amps {amplitudes.shape}")
    return amplitudes, phases

if __name__=='__main__':
    # Create parameters
    params = generate_parameters()
    centerPixel = (params["fieldOfViewBins"]//2, params["fieldOfViewBins"]//2)
    ## In the paper they use a detector exactly half-way between sources.
    #modulationFrequency = 0.2e9 #Hz - Can it be arbitrary??
    #modulationFrequency = 8e7 # First freq paper
    modulationFrequency = 6.5e8 # Second freq paper
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
    bin_delay = np.int((1/params["timeResolution"])*timeShift)
    beamInput = bGen.intensity_distribution([-1.5,0],
                                                 input_width, 9)
    print(f"Position of max: {np.unravel_index(np.argmax(beamInput), beamInput.shape)}")
    print(f"Beam padding: {beamInput.shape}")
    beamInput2 = bGen.intensity_distribution([1.5,0],
                                                 input_width, 9+bin_delay)
    print(f"Position of max: {np.unravel_index(np.argmax(beamInput2), beamInput2.shape)}")
    print(f"Beam padding: {beamInput2.shape}")
    beamInput += beamInput2 # Beams are 3cm apart from each other.
    #bGen.visualize(beamInput2)

    # Mask loading - absorber
    mask = np.genfromtxt('test_masks/tri.txt')
    mask = np.divide(mask,255)
    mask = np.ones([diffSim.FoVNumBins, diffSim.FoVNumBins])#mask.shape)
    objWidth=4
    mask[:,0:4] = np.zeros([33,objWidth])
    mask = np.pad(mask,(params["padSize"],)*2,'constant',constant_values=1)

    # Perform iterative simulation
    pl1 = params["propagationLen1"]
    pl2 = params["propagationLen2"]
    amplitudes, phases = iterative_roll_measurements(diffSim,
                                                     beamInput,mask,
                                                     modulationFrequency,
                                                     objWidth,pl1,pl2)
    print(phases.shape)

    """
    X_, Y_ = np.meshgrid(np.arange(phases.shape[0]),
                         np.arange(phases.shape[1]))
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X_,Y_, phases.transpose(), linewidth=0)
    ax.set_ylabel("X (cm) [Cross-section y=16 bin]")
    ax.set_xlabel("Scatterer Position (bins, from far left to far right)")
    ax.set_zlabel("Phase of nu_f at bucket-detector in coord [x,y=16]")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X_,Y_, amplitudes.transpose(), linewidth=0)
    ax.set_ylabel("X (cm) [Cross-section y=16 bin]")
    ax.set_xlabel("Scatterer Position (bins, from far left to far right)")
    ax.set_zlabel("Amplitude of nu_f at bucket-detector in coord [x,y=16]")
    """
    plt.subplot(1,2,1)
    plt.plot(amplitudes)
    plt.xlabel('Scatterer position in bins(left to right)')
    plt.ylabel('Amplitude of nu_f')
    plt.subplot(1,2,2)
    plt.plot(phases)
    plt.xlabel('Scatterer position in bins(left to right)')
    plt.ylabel('Phase of nu_f')

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
