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
from utils import generate_parameters

def bin_shift(nu_f, timeResolution):
    """args:
        nu_f: Modulation frequency
        timeResolution: Time interval between consecutive time bins (resolution)
    returns:
        Number of bins to shift to simulate time-delay
    """
    timeShift = 1/(2*nu_f)
    return np.int((1/timeResolution)*timeShift)
def combine_signal(s1, s2, nu_f, timeRes):
    """args:
      s1: First signal (time series of amplitude of else). Can be ndarray, where the last
        axis is the time dimension.
      s2: Second signal, to be time-shifted by t=1/2*nu_f. As above.
      nu_f: modulation frequency.
      timeRes: Time resolution paramter (in s), as in parameters
    returns:
      Combination of signal 1 and signal 2, ready for FFT.
    """
    binShift = bin_shift(nu_f, timeRes)
    s= s1+np.roll(s2, binShift)
    padding = (len(s.shape)-1)*((0,0),) + ((0,200),)
    sPadded = np.pad(s, padding) #we want to generalise this to add the padding to the
    # last  axis 
    return sPadded

def linquidst_signal_transform(signal, nu_f, ):
    return

def process_measurement(outInterface, nu_f, timeRes):
    """args:
        outInterface: The NxNxT array holding the measurements of the detector.
        nu_f: Modulation frequency
        timeRes: Time resolution paramter (in s), as in parameters
    returns:
        Combined signals of vertical (VDet) and horizontal (HDet) detector pairs.

    """
    # Compute Linquidst suggestion of 2D pattern of detectors by combining two conjugate
    # cells' signals by delaying the conjugated cell's signal. To start with it, we use
    # only one conjugate pair: equivalent to the original experiment of 2 sources.
    sep=17
    h = outInterface.shape[0]
    w = outInterface.shape[1]

    signalsVDetPairs = []
    signalsHDetPairs = []
    def measurements_accross_arr(l,measurements, horizontal=False):
        index_h_v = lambda x: (slice(None), x, slice(None)) if horizontal else\
            (x, slice(None),slice(None))
        signals=[]

        for i in range(l-sep):
           signal1 = measurements[index_h_v(i)]
           signal2 = measurements[index_h_v(i+sep)]
           signal = combine_signal(signal1,signal2,nu_f,timeRes)
           signals.append(signal)
        return np.array(signals)


    #VDetPair: Detector pairs vertically paired
    #HDetPair: Detector pairs horizontally paired
    signalsVDetPairs= measurements_accross_arr(h, outInterface, False)
    signalsHDetPairs= measurements_accross_arr(w, outInterface, True)
    print(f"Signals info -- V:{signalsVDetPairs.shape}, H:{signalsHDetPairs.shape}")

    #fftVSignals = np.fft.fft(signalsVDetPairs, axis=-1)
    #fftHSignals = np.fft.fft(signalsHDetPairs, axis=-1)
    #print(f"fft signals info -- V:{fftVSignals.shape}, H:{fftHSignals.shape}")


    """
    X = np.arange(signalsVDetPairs.shape[1])
    Y = np.arange(signalsVDetPairs.shape[-1])
    X,Y = np.meshgrid(X, Y)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X,Y, signalsVDetPairs[0].T, linewidth=0, antialiased=False)
    ax.set_zlabel('Signal Amplitude', rotation = 0)
    ax.set_ylabel('Time Bins', rotation = 0)
    ax.set_xlabel('Spatial location', rotation = 0)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X,Y, signalsVDetPairs[-1].T, linewidth=0, antialiased=False)
    ax.set_zlabel('Signal Amplitude', rotation = 0)
    ax.set_ylabel('Time Bins', rotation = 0)
    ax.set_xlabel('Spatial location', rotation = 0)

    plt.show()
    """

    return (signalsVDetPairs, signalsHDetPairs)


def get_nearest_freq_el(nu_f, frequencies):
    """args:
    returns:
        The index of the frequency nearest to the modulation frequency.
    """
    # DFT gives bins (sampling), so we need to find the nearest bin
    return np.argmin(np.abs(frequencies-nu_f))

def iterative_roll_measurements(diffSim, beam, iniMask, nu_f, objWidth, pLen1,pLen2,
                                sourceSeparation):
    """args:
        diffSim:
        beam:
        iniMask:
        nu_f:
        objWidth:
        pLen1:
        pLen2:
        sourceSeparation:
    return:
        Scatterer position vs nu_f amplitude and phase

    """
    nIterations = diffSim.FoVNumBins - objWidth
    mask = iniMask
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
        #mask = np.roll(mask,1)
        beam = np.roll(beam,1, axis=1)
        #signal = np.pad(outInterface, ((0,),(0,),(200,)))
        signalsV, signalsH = process_measurement(outInterface, nu_f, diffSim.timeRes)
        ftAmplitudesV = np.fft.fftn(signalsV, axes=[-1])
        ftAmplitudesH = np.fft.fftn(signalsH, axes=[-1])

        ftFreq = np.fft.fftfreq(ftAmplitudesV.shape[-1], diffSim.timeRes)
        ftFreq = np.fft.fftshift(ftFreq)

        ftAmplitudesV = np.fft.fftshift(ftAmplitudesV, axes=[-1])
        ftAmplitudesH = np.fft.fftshift(ftAmplitudesH, axes=[-1])
        nearestFreq = get_nearest_freq_el(nu_f, ftFreq)

        ftAmplitudesV = ftAmplitudesV[:,:,nearestFreq]
        ftAmplitudesH = ftAmplitudesH[:,:,nearestFreq].T

        ftPhasesV = np.angle(ftAmplitudesV)
        ftAmplitudesV = np.abs(ftAmplitudesV)

        ftPhasesH = np.angle(ftAmplitudesH)
        ftAmplitudesH = np.abs(ftAmplitudesH)


        print(f"V ftAmplitudes and Phases: {ftAmplitudesV.shape}, {ftPhasesV.shape}")
        print(f"H ftAmplitudes and Phases: {ftAmplitudesH.shape}, {ftPhasesH.shape}")

        if  i%9==0:
            oi = np.sum(outInterface,-1)
            oi[:,0]=1
            oi[:,10]=1
            mi = np.sum(middleInterface,-1)
            plt.figure()
            plt.imshow(oi, interpolation='none')
            plt.figure()
            plt.imshow(np.sum(middleInterface,-1),interpolation='none')
            plt.show()

            plt.figure()
            plt.imshow(ftPhasesV)
            plt.title('Image of phases accross array (Vertical detectors)')
            plt.figure()
            plt.imshow(ftAmplitudesV)
            plt.title('Image of amplitudes accross array (Vertical detectors)')
            plt.figure()
            plt.imshow(ftPhasesH)
            plt.title('Image of phases accross array (Horizontal detectors)')
            plt.figure()
            plt.imshow(ftAmplitudesH)
            plt.title('Image of phases accross array (Horizontal detectors)')

            plt.show()

            # What axis is the row position and which one is the column position?
            # The axis with reduced length is the axis along which the convolution
            # happens, so the axis with the non-reduced length is the axis where coupled
            # detectors are taken.
            """
            X = np.arange(ftAmplitudesV.shape[1])
            Y = np.arange(ftAmplitudesV.shape[0])
            X,Y = np.meshgrid(X, Y)
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(X,Y, ftAmplitudesV, linewidth=0, antialiased=False)
            ax.set_zlabel('nu_f Amplitude (FFT)', rotation = 0)
            ax.set_ylabel('Spatial Y (Column bin)', rotation = 0)
            ax.set_xlabel('Spatial X (Row bin)', rotation = 0)

            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(X,Y, ftAmplitudesH, linewidth=0, antialiased=False)
            ax.set_zlabel('nu_f Amplitude (FFT)', rotation = 0)
            ax.set_xlabel('Spatial Y (Column bin)', rotation = 0)
            ax.set_ylabel('Spatial X (Row bin)', rotation = 0)
            plt.show()

            #plt.savefig(directory+f"absorber_{i+1}.png")
            """

        amplitudes.append(ftAmplitudesV)
        phases.append(ftPhasesV)
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
    maskp = np.genfromtxt('test_masks/tri.txt')
    maskp = np.divide(mask,255)
    mask = np.ones([diffSim.FoVNumBins, diffSim.FoVNumBins])#mask.shape)
    mask[1:,1:,:] = maskp

    #objWidth=4
    #mask[:,:objWidth] = np.zeros([33,objWidth])
    mask = np.pad(mask,(params["padSize"],)*2,'constant',constant_values=1)

    # Perform iterative simulation
    pl1 = params["propagationLen1"]
    pl2 = params["propagationLen2"]
    amplitudes, phases = iterative_roll_measurements(diffSim,
                                                     beamInput,mask,
                                                     modulationFrequency,
                                                     objWidth,pl1,pl2, d_from_center*2)
    np.savetxt("amplitudes.csv",amplitudes, delimiter=",")
    np.savetxt("phases.csv",phases, delimiter=",")
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
