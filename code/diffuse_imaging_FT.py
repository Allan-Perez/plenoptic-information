"""
Created on Thu Aug 30 10:28:53 2018
@author: Ashley Lyons, School of Physics and Astronomy, Univeristy of Glasgow, ashley.lyons@glasgow.ac.uk

Functions
"""
probe_coord = (16,16)

# Point Spread Function from propagation through slab of diffuse medium - Eq.2 in paper
def _PSF_eq(r_sq,t):
    return c/((4*math.pi*D*c*t)**1.5) * \
            np.exp(-r_sq/(4*D*c*t)) * np.exp(-mu_a*c*t)

def point_spread_function(propagation_length):
    # Vectorized PSF computation in the NxN grid at cross-section z=propagation_length
    im1 = np.tile(np.square(x),[FoV_n_bins,1])
    im2 = np.transpose(np.tile(np.square(y),[FoV_n_bins,1]))
    im = im1 + im2
    r_sq = im + np.square(propagation_length)
    PSF = np.zeros(a)

    t_bins = t_res*(np.arange(t_n_bins)+1)
    PSFaux = np.zeros([FoV_n_bins,FoV_n_bins, t_n_bins]) + r_sq[:,:,None]
    vPSFeq = np.vectorize(_PSF_eq)
    PSFaux = vPSFeq(PSFaux, t_bins)

    PSF[xypad:(xypad+FoV_n_bins),xypad:(xypad+FoV_n_bins),:]= PSFaux

    return PSF

# Convolution with PSF via FFT    
def diff_conv(Phi_in,PSF):
    ## ALLAN: Question - What does it mean to convolve PSF and Phi? 
    ## I think it may mean that it's the way to propagate the input beam??
    ## since convolution is just the integration of the product as one 
    ## function moves from left to right.
    print(Phi_in.shape, PSF.shape)


    Phi = np.real((np.fft.ifftn(np.multiply(np.fft.fftn(Phi_in),np.fft.fftn(PSF)))))
    Phi = np.divide(Phi,np.amax(Phi))   # normalise

    """
    plt.subplot(1, 3, 1)
    plt.plot(Phi_in[probe_coord],label="Input beam")
    plt.xlabel('Time bins')
    plt.ylabel('Value of Phi of input impulse')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(PSF[probe_coord],label="PSF")
    plt.xlabel('Time bins')
    plt.ylabel('Value of PSF(??)')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(Phi[probe_coord],label="Phi Convolved input impulse into PSF")
    plt.xlabel('Time bins')
    plt.ylabel('Convolved PSF with input')
    plt.legend()
    plt.show()
    """

    return Phi

# Full forward model including propagation through both slabs and object masking
def forward_model(b_input,mask):
    # calculate propagation through first slab
    PSF1 = point_spread_function(propagation_length1)

    Phi = diff_conv(b_input,PSF1)
    Phi = np.fft.fftshift(Phi) # FRANCESCO: Fixing frequency offset issue

    # hidden object masking
    for tt in range(t_n_bins):
        Phi[:,:,tt] = np.multiply(Phi[:,:,tt],mask)

    # only recalculate PSF for second slab if needed
    Phi_m = Phi+0 # FRANCESCO: Just an extra output to visualise the field in the middle
    ## ALLAN: Question - PL1 and PL2 are just lenghts, not coordinates. If they have
    ## the same value, it doesn't imply that the second slab should have "0 length".
    if propagation_length2 == propagation_length1:
        Phi = diff_conv(Phi,PSF1)
        Phi = np.fft.fftshift(Phi) # FRANCESCO: Fixing frequency offset issue
    else:
        PSF2 = point_spread_function(propagation_length2)
        Phi = diff_conv(Phi,PSF2)
        Phi = np.fft.fftshift(Phi) # FRANCESCO: Fixing frequency offset issue

    # Crop data only if padded with 0s
    if xypadding != FoV_n_bins:

        Phi = Phi[
            np.int(xypadding/2-FoV_n_bins/2)-1:np.int(xypadding/2+FoV_n_bins/2)-1,
            np.int(xypadding/2-FoV_n_bins/2)-1:np.int(xypadding/2+FoV_n_bins/2)-1,:]
#        Phi_m = Phi_m[np.int(xypadding/2-FoV_n_bins/2)-1:np.int(xypadding/2+FoV_n_bins/2)-1,np.int(xypadding/2-FoV_n_bins/2)-1:np.int(xypadding/2+FoV_n_bins/2)-1,:]

    return Phi, Phi_m # FRANCESCO: Extra output - field in the middle

"""
Initial parameters
"""
import numpy as np, matplotlib.pyplot as plt, math, time

n = 1.44                   # refractive index of diffuse medium
propagation_length1 = 2.5  # thickness of first diffuse medium (cm)
propagation_length2 = 2.5  # thickness of second diffuse medium (cm)

mu_a = 0.09     # absorption coefficient (cm^-1)
mu_s = 16.5     # scattering coefficient (cm^-1)
c = 3e10/n      # phase velocity in medium (cm/s)

FoV_length = 4.4     # size of the camera field of view (cm)
t_res = 55e-12       # camera temporal bin width (ps)
t_n_bins = 251       # number of temporal bins
FoV_n_bins = 32      # number of camera pixels

x = np.linspace(-FoV_length/2,FoV_length/2,FoV_n_bins)       # array of positionsin FoV (cm)
y = np.linspace(-FoV_length/2,FoV_length/2,FoV_n_bins)       # array of positionsin FoV (cm)

D = (3*(mu_a+mu_s))**(-1)            # "D" as defined in paper

xypadding = 64                  # Pixel count after 0 padding for the FFTs - avoids wrapping
xypad = int((xypadding - FoV_n_bins)/2)

"""
Input intensity distribution - Gaussian
"""
input_center = [0,0]        # beam center position (cm)
input_width = 5           # beam width (cm, s.d.)
pulse_start_bin = 9         # starting bin for the input pulse
a = [xypadding,xypadding,t_n_bins]
b_input = np.zeros(a)
for xx in range(FoV_n_bins):
    for yy in range(FoV_n_bins):
        b_input[xx+xypad,yy+xypad,pulse_start_bin] = \
            np.multiply(
                math.exp(-np.square( (x[xx]-input_center[0]) )/input_width**2 ),
                math.exp(-np.square((y[yy]-input_center[1]))/input_width**2))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X = np.arange(FoV_n_bins)
Y = np.arange(FoV_n_bins)
X,Y = np.meshgrid(X,Y)
surf = ax.plot_surface(X, Y, \
                       b_input[xypad:xypad+FoV_n_bins,xypad:xypad+FoV_n_bins,9], 
                       linewidth=0, antialiased=False)
plt.show()




"""
Test parameters
"""
# load test mask
#mask = np.genfromtxt('C:/Users/Trial/Documents/MATLAB/Diffuse_imaging_AL/test_masks/tri.txt')
mask = np.genfromtxt('test_masks/tri.txt')
mask = np.divide(mask,255)

mask = np.pad(mask,(xypad,xypad),'constant',constant_values=1)


# function profiling
ts = time.time()
test, test_m = forward_model(b_input,mask)
#test = point_spread_function(2.50)
tf = time.time()
print(tf-ts)

plt.subplot(1, 3, 1)
plt.imshow(mask,interpolation='none') # FRANCESCO: Visualise Mask
plt.title("Hidden Object")
plt.subplot(1, 3, 2)
plt.imshow(np.sum(test_m,2),interpolation='none') # FRANCESCO: Visualise Field in the Middle
plt.title("Middle Surface")
plt.subplot(1, 3, 3)
plt.imshow(np.sum(test,2),interpolation='none')
plt.title("End Surface")
plt.show()
