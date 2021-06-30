# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 10:28:53 2018
@author: Ashley Lyons, School of Physics and Astronomy, Univeristy of Glasgow, ashley.lyons@glasgow.ac.uk

Functions
"""
from matplotlib import  pyplot as plt

# Point Spread Function from propagation through slab of diffuse medium - Eq.2 in paper
def point_spread_function(propagation_length):
    PSF = np.zeros(a)
    for tt in range(t_n_bins):
        t = (tt+1)*t_res    # time at each step (ps, arbitrary start)
        # calculate matrix of x^2 + y^2 values - avoids for loop in spatial dimensions
        im1 = np.tile(np.square(x),[FoV_n_bins,1])
        im2 = np.transpose(np.tile(np.square(y),[FoV_n_bins,1]))
        im = im1 + im2
        # Eq.2 in paper
        PSF[xypad:(xypad+FoV_n_bins),xypad:(xypad+FoV_n_bins),tt] = c/((4*math.pi*D*c*t)**1.5) * np.exp(-(im + np.square(propagation_length))/(4*D*c*t)) * np.exp(-mu_a*c*t)
        plt.plot(PSF[16,16,:])
    return PSF

# Convolution with PSF via FFT    
def diff_conv(Phi_in,PSF):
    Phi = np.real((np.fft.ifftn(np.multiply(np.fft.fftn(Phi_in),np.fft.fftn(PSF)))))
    Phi = np.divide(Phi,np.amax(Phi))   # normalise
    return Phi

# Full forward model including propagation through both slabs and object masking
def forward_model(b_input,mask):
    # calculate propagation through first slab
    PSF1 = point_spread_function(propagation_length1)
    PSF1[PSF1<700000] = 0 # FRANCESCO: Thresholding to avoid square windowing. Should really compute the PSF over a larger area instead.
    Phi = diff_conv(b_input,PSF1)
    Phi = np.fft.fftshift(Phi) # FRANCESCO: Fixing frequency offset issue

    # hidden object masking
    for tt in range(t_n_bins):
        Phi[:,:,tt] = np.multiply(Phi[:,:,tt],mask)

    # only recalculate PSF for second slab if needed
    Phi_m = Phi+0 # FRANCESCO: Just an extra output to visualise the field in the middle
    if propagation_length2 == propagation_length1:
        Phi = diff_conv(Phi,PSF1)
        Phi = np.fft.fftshift(Phi) # FRANCESCO: Fixing frequency offset issue
    else:
        PSF2 = point_spread_function(propagation_length2)
        Phi = diff_conv(Phi,PSF2)
        Phi = np.fft.fftshift(Phi) # FRANCESCO: Fixing frequency offset issue

    # Crop data only if padded with 0s
    if xypadding != FoV_n_bins:

        Phi = Phi[np.int(xypadding/2-FoV_n_bins/2)-1:np.int(xypadding/2+FoV_n_bins/2)-1,np.int(xypadding/2-FoV_n_bins/2)-1:np.int(xypadding/2+FoV_n_bins/2)-1,:]
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
input_width = 0.5           # beam width (cm, s.d.)
pulse_start_bin = 9         # starting bin for the input pulse
a = [xypadding,xypadding,t_n_bins]
b_input = np.zeros(a)
for xx in range(FoV_n_bins):
    for yy in range(FoV_n_bins):
        b_input[xx+xypad,yy+xypad,pulse_start_bin] = np.multiply(math.exp(-np.square((x[xx]-input_center[0]))/input_width**2),math.exp(-np.square((y[yy]-input_center[1]))/input_width**2))

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

