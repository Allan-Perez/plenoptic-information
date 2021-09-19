"""
Created on Thu Aug 30 10:28:53 2018
@author: Ashley Lyons, School of Physics and Astronomy, Univeristy of Glasgow, ashley.lyons@glasgow.ac.uk

Modified on Tue Jul 6 13:30 2021
@author: Allan Perez, School of Physics and Astronomy, University of Glasgow,
2480326p@student.gla.ac.uk
The modification consists of a refactoring into OOP for easy multiparam simulations, and
avoid having global variables or having to pass params around functions.
"""
import numpy as np, matplotlib.pyplot as plt
class DiffusionSim():
    """Diffusion simulator. Takes in the needed physical and grid parameters,
       and computes the diffusion approx according the point spread function (PSF) found
       in the paper. The simulation consists of a grid NxNxT where N is the number of
       spatial bins used to measure, in our case the SPAD camera, and T is the number of
       temporal A bins.

       The PSF is the solution to the impulse input (delta function), i.e. a green's
       functions, and the solution to an arbitrary input shape is the convolution between
       the PSF and that arbitrary input shape. (See Green's Function in wikipedia for more
       details.
    """
    def __init__(self, **kwargs):
        # Grid Params
        self.tBinsN = kwargs["timeNumBins"]
        self.FoVNumBins = kwargs["fieldOfViewBins"]
        self.xCoordSpace = kwargs["xCoordSpace"]
        self.yCoordSpace = kwargs["yCoordSpace"]
        self.padSize = kwargs["padSize"]
        self.xyPadding = 2*self.padSize+self.FoVNumBins

        # Grid Auxiliar
        self.FoVGridDims = [self.FoVNumBins,self.FoVNumBins, self.tBinsN]
        self.paddedGridDims = [self.xyPadding,self.xyPadding,self.tBinsN]

        # Physics Params
        self.c = kwargs["c"] # Phase velocity in medium (cm/s)
        self.D = kwargs["D"] # Diffusion coefficient
        self.mu_a = kwargs["mu_a"] # Absorption coefficient
        self.timeRes = kwargs["timeResolution"]


    def _PSF_eq(self, r_sq, t):
        """Point Spread Function from propagation through slab of
           diffuse medium - Eq.2 in paper
        """
        return self.c/((4*np.pi*self.D*self.c*t)**1.5) * \
                np.exp(-r_sq/(4*self.D*self.c*t)) * np.exp(-self.mu_a*self.c*t)

    def PSF (self, propagation_length):
        """ Vectorized PSF computation in the NxNxT grid at
            cross-section z=propagation_length.
        """
        im1 = np.tile(np.square(self.xCoordSpace),[self.FoVNumBins,1])
        im2 = np.transpose(np.tile(np.square(self.yCoordSpace),[self.FoVNumBins,1]))
        im = im1 + im2
        r_sq = im + np.square(propagation_length)

        # Vectorize Eq.2 PSF, and broadcast along all t_bins,
        # to generate the R^NxNxT matrix/tensor representing PSF
        PSF = np.zeros(self.paddedGridDims)
        tBins = self.timeRes*(np.arange(self.tBinsN)+1)
        PSFaux = np.zeros(self.FoVGridDims) + r_sq[:,:,None]
        vPSFeq = np.vectorize(self._PSF_eq) # Careful - self!
        PSFaux = vPSFeq(PSFaux, tBins)

        # PSF with padding
        xyIni, xyEnd = (self.padSize, self.padSize+self.FoVNumBins)
        PSF[xyIni:xyEnd,xyIni:xyEnd,:]= PSFaux
        print(f'Number of time bins: {self.tBinsN}')
        return PSF

    def convolution(self, Phi_in, PSF):
        """Green's Function theorem for solving inhomogeneous
        boundary value problems. Convolution via FFT."""

        Phi = np.real((np.fft.ifftn(
            np.multiply(np.fft.fftn(Phi_in),np.fft.fftn(PSF))
        )))
        # Normalise only in space
        #Phi = np.fft.fftshift(np.divide(Phi,np.amax(Phi)), axis=(0,1))
        Phi = np.fft.fftshift(np.divide(Phi,np.amax(Phi)))
        #Phi = np.divide(Phi-Phi.mean(),Phi.std())#np.amax(Phi))   # normalise
        #Phi = np.divide(Phi,Phi.max())#np.amax(Phi))   # normalise
        # fftshift causes bugs in the time of arrival of signals.
        """ --- DEBUGING
        print(f'Stats of computed phi: {Phi.mean(), Phi.std()}')
        print(f'Stats of in phi: {Phi_in.mean(), Phi_in.std()}')
        print(f'Total value of phi: {Phi.flatten().sum()}')

        plt.figure()
        plt.plot(Phi_in[15+16,:].T)
        plt.title("Computed Phi, looking for bug")
        plt.figure()
        plt.plot(PSF[15+16,:].T)
        plt.title("Computed PSF, looking for bug")
        plt.figure()
        plt.plot(Phi[15+16,:].T)
        plt.title("Convoluted, looking for bug")
        plt.show()
        """

        return Phi # FRANCESCO: Fixing frequency offset issue

    def forward_model(self, beamInput, mask, propagationLen1, propagationLen2):
        """Full forward model including propagation through
        both slabs and object masking.
        """
        # Calculate propagation through first slab
        PSF1 = self.PSF(propagationLen1)
        Phi = self.convolution(beamInput, PSF1)

        # Hidden object masking - across the whole time axis (broadcasting)
        Phi *= mask[:,:,None]

        # Recalculate PSF for second slab if needed. 
        # Store middle cross-section
        Phi_m = Phi.copy()
        if propagationLen1 == propagationLen2 :
            print("Plotting out surface")
            Phi = self.convolution(Phi,PSF1)
        else:
            PSF2 = self.PSF(propagationLen2 )
            Phi = self.convolution(Phi,PSF2)

        # Crop data only if padded with 0s
        if self.padSize!=0:
            Phi = Phi[
                self.padSize-1:self.padSize+self.FoVNumBins-1,
                self.padSize-1:self.padSize+self.FoVNumBins-1,:]
            print(f"Cropping data: from {self.padSize} to {self.padSize+self.FoVNumBins}")

        return Phi, Phi_m # FRANCESCO: Extra output - field in the middle

class BeamGenerator():
    def __init__(self, **kwargs):
        self.dists = {
            "gaussian": lambda x,mu,sig: np.exp(-np.square( (x-mu) /sig ) )
        }
        self.FoVNumBins = kwargs["fieldOfViewBins"]
        self.padSize = kwargs["padSize"]
        self.xCoordSpace = kwargs["xCoordSpace"]
        self.yCoordSpace = kwargs["yCoordSpace"]
        self.tBinsN = kwargs["timeNumBins"]
        self.xyPadding = 2*self.padSize+self.FoVNumBins
        # Grid Auxiliar
        self.FoVGridDims = [self.FoVNumBins,self.FoVNumBins, self.tBinsN]
        self.paddedGridDims = [self.xyPadding,self.xyPadding,self.tBinsN]

    # Compute the input beam shape with meshgrid
    def intensity_distribution(self, inputCenter, inputWidth,
                               pulseStartBin, visual=False, returnUnpadded=False):
        beamInput=np.zeros(self.paddedGridDims)
        dist = self.dists["gaussian"]

        X_, Y_ = np.meshgrid(self.xCoordSpace,self.yCoordSpace)
        intensityShape = \
            dist(X_,inputCenter[0], inputWidth)*\
            dist(Y_,inputCenter[1], inputWidth)
        xyIni, xyEnd = (self.padSize+1,self.FoVNumBins+self.padSize+1)
        beamInput[xyIni:xyEnd,xyIni:xyEnd,pulseStartBin] = intensityShape
        print(f"Padded beams max: {np.unravel_index(beamInput.argmax(), beamInput.shape)}")

        if visual:
            self.visualize(intensityShape)
        if returnUnpadded:
            return (beamInput,intensityShape)

        return beamInput
    def visualize(self, beam):
        X_, Y_ = np.meshgrid(self.xCoordSpace,self.yCoordSpace)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X_,Y_,beam, linewidth=0, antialiased=False)
        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)")
        ax.set_zlabel("Beam Intensity")
        plt.show()

"""
Initial parameters
"""
if __name__=="__main__":
    import time
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

    diffSim = DiffusionSim(**params)

    """
    Input intensity distribution - Gaussian
    """
    bGen = BeamGenerator(**params)

    input_center = [0,0]        # beam center position (cm)
    input_width = 0.5           # beam width (cm, s.d.)
    pulse_start_bin = 9         # starting bin for the input pulse
    #beamInput  = bGen.intensity_distribution([-1,0], input_width, 9, True)
    #beamInput2 = bGen.intensity_distribution([1,0],  input_width, 9, True)
    #beamInput += beamInput2
    beamInput = bGen.intensity_distribution(input_center,input_width, 9,True)

    """
    Test parameters
    """
    # load test mask
    mask = np.genfromtxt('test_masks/tri.txt')
    mask = np.divide(mask,255)
    mask = np.pad(mask,(params["padSize"],)*2,'constant',constant_values=1)

    # function profiling
    ts = time.time()
    pl1 = params["propagationLen1"]
    pl2 = params["propagationLen2"]
    test, test_m = diffSim.forward_model(beamInput,mask,pl1,pl2)
    tf = time.time()
    print(tf-ts)

# Time-Integrated Visualization 
    nim=3
    plt.subplot(1, nim, 1)
    plt.imshow(mask,interpolation='none')
    plt.title("Hidden Object")
    plt.subplot(1, nim, 2)
    plt.imshow(np.sum(test_m,2),interpolation='none')
    plt.title("Middle Surface")
    plt.subplot(1, nim, 3)
    plt.imshow(np.sum(test,2),interpolation='none')
    plt.title("End Surface")
    plt.show()
