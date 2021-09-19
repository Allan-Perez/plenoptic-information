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
        nu_f:
        timeResolution:
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

def process_measurement(outInterface, nu_f, timeRes):
    """args:
        outInterface: 
        nu_f:
        timeRes: 
    returns:
        Combined signals of vertical (VDet) and horizontal (HDet) detector pairs.

    """
    # Compute Linquidst suggestion of 2D pattern of detectors by combining two conjugate
    # cells' signals by delaying the conjugated cell's signal. To start with it, we use
    # only one conjugate pair: equivalent to the original experiment of 2 sources.
    sep=10
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


    signalsVDetPairs= measurements_accross_arr(h, outInterface, False)
    signalsHDetPairs= measurements_accross_arr(w, outInterface, True)
    print(f"Signals info -- V:{signalsVDetPairs.shape}, H:{signalsHDetPairs.shape}")

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

def informationProcessing(X, Y):
    """args:
        X: Message sent. N symbols flat array.
        Y: Message received. N symbolts flat array.
    """

    return

def encode_message(signal):
    return


def signalToSymbol(X):
    """args:
        X: NxNxT array where each element is a time series (hence last axis is time).
    returns:
        NxN (flatten array) encoded message into discrete set of symbols represented by
        integers. Maximum number of symbols are n*t, where n is the number of amplitude
        bins and t are the number of t bins.
    """
    signals = {
        'r': X.max(-1).flatten() ,
        't': X.argmax(-1).flatten(),
    }
    #print(X.max(-1).shape, X.max(-1)[-1])
    plt.figure()
    t= signals['t']
    r= signals['r']

    #message = encode_message(t)#,r)


    title = "Sent message signals" if r[r==0].size>0 else "Received message signals"
    plt.scatter(t[t!=0],r[r!=0] )
    plt.title(title)
    plt.xlabel('Time of first max')
    plt.ylabel('Amplitude of first max')

    plt.figure()
    ns, bins, _= plt.hist(r)
    print(f'Bins for r: {bins}')
    #print(f'Bins for r: ')

    plt.show()

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
        mask = np.roll(mask,1)
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
        ftAmplitudesH = ftAmplitudesH[:,:,nearestFreq]

        ftPhasesV = np.angle(ftAmplitudesV)
        ftAmplitudesV = np.abs(ftAmplitudesV)

        ftPhasesH = np.angle(ftAmplitudesH)
        ftAmplitudesH = np.abs(ftAmplitudesH)

        print(f"V ftAmplitudes and Phases: {ftAmplitudesV.shape}, {ftPhasesV.shape}")
        print(f"H ftAmplitudes and Phases: {ftAmplitudesH.shape}, {ftPhasesH.shape}")

        messageSignal = middleInterface[15:47,15:47,:]
        receivedSignal = outInterface
        plt.figure()
        #plt.plot(np.log(messageSignal[26,24,:]), label='Sent')
        #plt.plot(np.log(receivedSignal[26,24,:]), label='Received')
        plt.plot(messageSignal[23,:,:].T, label='Sent')
        plt.legend()
        plt.title('Sent signals, of detectors in the column 23')
        plt.figure()
        plt.plot(receivedSignal[23,:,:].T, label='Received')
        plt.title('Received signals, of detectors in the column 23')
        plt.legend()
        plt.show()
        messageSymbols = signalToSymbol(messageSignal)
        m = signalToSymbol(receivedSignal)

        informationProcessing(messageSignal, receivedSignal)



        if   i%9==0:
            oi = np.sum(outInterface,-1)
            #oi[:,0]=1
            #oi[:,10]=1
            mi = np.sum(middleInterface,-1)
            plt.figure()
            plt.imshow(oi, interpolation='none')
            plt.figure()
            plt.imshow(np.sum(middleInterface,-1),interpolation='none')
            plt.show()

            # What axis is the row position and which one is the column position?
            # The axis with reduced length is the axis along which the convolution
            # happens, so the axis with the non-reduced length is the axis where coupled
            # detectors are taken.
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

        amplitudes.append(ftAmplitudesV)
        phases.append(ftPhasesV)
        #amplitudes.append(np.abs(ftAmplitudeMidPixel[nearestFreq]))
        #phases.append(np.angle(ftAmplitudeMidPixel[nearestFreq]))
    amplitudes, phases = (np.array(amplitudes),np.array(phases))
    print(f"Shapes of time-bins amps {amplitudes.shape}")
    return amplitudes, phases

if __name__=='__main__':
    # Create parameters
    params = generate_parameters(32, 64)
    # Even number of bins required for message sizes of same size
    #params['fieldOfViewBins'] = 32
    #params['padSize'] = (64-32)//2
    modulationFrequency = 0.2e9 #8e7, 6.5e8 Hz,  - Can it be arbitrary??

    # Simulators
    diffSim = DiffusionSim(**params)
    bGen = BeamGenerator(**params)

    # Beam shape
    # Beam's note: It is supposed to be equidistant from the two sources. 
    # So in any case, the detector should be at [0,0]
    input_center = [0,0]        # beam center position (cm)
    input_width = 0.5           # beam width (cm, s.d.)
    beamInput = bGen.intensity_distribution([0,0], input_width, 9)
    print(f"Position of max: {np.unravel_index(np.argmax(beamInput), beamInput.shape)}")
    print(f"Beam padding: {beamInput.shape}")

    # Mask loading - absorber
    mask = np.genfromtxt('test_masks/tri.txt')
    mask = np.divide(mask,255)
    mask = np.ones([diffSim.FoVNumBins, diffSim.FoVNumBins])#mask.shape)
    objWidth=4
    mask[:,:objWidth] = np.zeros([32,objWidth])
    mask = np.pad(mask,(params["padSize"],)*2,'constant',constant_values=1)

    # Perform iterative simulation
    pl1 = params["propagationLen1"]
    pl2 = params["propagationLen2"]
    amplitudes, phases = iterative_roll_measurements(diffSim,
                                                     beamInput,mask,
                                                     modulationFrequency,
                                                     objWidth,pl1,pl2, 0)
