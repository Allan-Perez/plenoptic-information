# Plenoptic information for diffuse imaging
This is the repo for the University of Glasgow's summer project from the physics
department, the extreme light laboratory.
## What is "Plenoptic"?
The word plenoptic refers to the idea of a "light field", which carries the information of
amount and direction of light travelling in every direction in a space. This space of all
possible light rays is a 5D-plenoptic function, where the magnitude of each ray is given
by radiance.
### 5D plenoptic function
When the modelled light is incoherent and interacting objects are much greater than the
wavelenght, the regime we work with is geometric optics, where the fundamental carrier is
a light ray. Magnitude of the object is referred to as _amount of light_ or _radiance_
(Dimensions of power per solid angle per area). The solid angle is a measure of how large
an object appears to an observer at a particular point. E.g. a star from earth looks
small, (smalled solid angle) while the sun looks huge (larger solid angle).

The term pleptonic function was coined by Adelson (1991) and it refers to the description
of radiance along all rays in 3D-space. Each ray can be parametrized by xyz-coordinates
and $\theta$,$\phi$ angles, and hence forming a 5D function over a 5D manifold, equivalent
to the product of $\mathbb{R}^3$ and the 2-sphere (tensor product of the two vector
spaces). The angle information is important, since the xyz coordinates only give
information about a "point in the line" (ray instead of line), and so we need information
about the direction of travel in the 3D euclidean space.

This is just one way to model plenoptic or light field, but there're more way than this.
One can consider a vector field in R3, or one could even consider polarization angle or
wavelength to construct and even higher dimensional function. Another popular way to model
this is the 4D light field, used in computer graphics, dubbed the photic field (Moon,
1981) that defines the direction of travel of a ray plus the radiance scalar.
### What are plenoptic (light field) cameras?
Plenoptic cameras attempts to capture the information of the light field that emanates
from a landscape/scene. The field is modelled by the direction of light rays travelling in
space plus the intensity of the ray. In constrast, conventional cameras only captures
light intensity. A design of plenoptic camera uses an array of micro-lenses in front of an
otherwise conventional image sensor, with the aim of sensing/capturing information of
intensity, color, and _directional_ information.

The main commercial application of this kind of camera is that a picture can be refocused
after they are taken. I ignore the usefuleness of this feature. The laboratory
## What is "diffuse imaging"?
The diffuse theory of light is an approximation to the behaviour of the light field as it
travels through scattering media. Apparently it can be derived from quantum mechanics
(find citation in Ripoll) but the most accessible derivation is from Poynting's theorem
and the Radiative Transfer Equation, ending up in a differential equation that resembles
Fourier's theory of diffusion of heat. The regime where analytical solutions to the
diffuse equation works is rather limited, but generally it assumes measurements in the
far-field, a scattering coefficient $mu_s$ much greater than the absorption coefficient
$mu_a$, and requires a homogeneous media. To study real geometries  Monte Carlo methods
are often used, although for analytical solutions are preferred. For instance, taking
advantage of Green's functions method for solving differential equation, it makes it easy
to simulate arbitrary incident light shapes and arbitrary geometries. 
## What is information theory?
It's the study of quantification and transmission of information. The amount of
information of a system depends on how many states it can have (the more states, usually
the more information it will have), and the probability distributions of these states
(when a few states are a lot more probable than the rest, it has not much information.
When the states are all almost equiprobable, the system has almost maximal information).
The amount of information of a system (or entropy) can be seen as synonim of the amount of
uncertainty of the state of that system. When there's a lot of uncertainty (all the states
can be equally probable) then there's a lot of information you can't really (or it's
really hard to) predict which one state is going to come next. When you have a lot of
certainty in the state (say, tomorrow is going to be sunny in Buenos Aires) then there's
little information in such a statement. 

The quantification of information helps to describe the amount of redundancy in a system
conveying a message, and hence helps to come up with ways of reducing this uncertainty
(Source coding theorem) and also helps to come up with ways of retrieving information in
uncertain scenarios (noisy channel theorem).

## What is the goal?
The goal is to determine from an information theory point of view whether gathering extra
information about the scattering of light would help reconstruct the scattered object.

In a more formal way, what we want to find is $I(X;Y)$ compared to $I(X;Y')$ where $X$ is
the mask or goal we're trying to infer, $Y$ is the ensemble of the
intensity received, and $Y'$ is the ensemble of the __incident angle__?? Recall
$$I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$
Another way of looking at this is whether $I(X;U)>I(X;Y)$ where $U=Y+Y'$

This inverse problem can be seen as a communication problem, where the noisy channel is
the diffusive material (following the diffusion equation) and  we're trying to
reconstruct a message that has been transmitted through that channel (an NxN array rather
than a Nx1 array). This channel is non-trivial in the sense that it can be viewed as in
the message being sent through the time-integrated intensity, or time-thresholded
intensity, or maybe there's a way of retrieving information from the shifts of intensity
through time (perhaps using close cross-sections and compute the gradient through space of
the diffusion (isn't that given by the equation?), and that's what we're interested in in
this project: is there information transmitted through other physical quantities?
k-vector? wavefront? phase of received signal?)

### What's the problem with wavefront?
I'm not too sure "measuring the wavefront of diffuse light" makes much sense, since the
whole point of diffuse light is it's measured as a scalar field which does not oscillate
(so totally incoherent), so there's no front with equal phase, not even with equal
direction. The diffuse equation doesn't even use the poynting vector, rather only the
intensity or flux. It describes that each point in space at each point in time will have a
specific curve of intensity of light. So I don't see a "wave-front" object here. 

However, we may be able to attack this from another angle: What if we define a "pseudo
wavefront shape" as the curve in space where the intensity is at peak at any given time.
Does that curve (or rather curved surface) carry information about the message (the
mask?).  And if so, how could we retrieve that with minimal loss of information? Does the
shape or number of sources of distance between sources (in space and in time) affect this
information (i.e. does using more sources at different locations in space and in time
increase the entropy of the channel?) The channel does physically depend not only in the
diffuse light equation, which we don't control, but also in the pulse shape, frequency and
wavelength, and the configuration (number and arrangement of sources in space and time),
which we control.

An interesting research point would be to completely model this as a communication channel
as follows:
$$ W \to f_n \to X^{NxNxT}\to p(y|x) \to Y^{NxNxT} \to g_n \to \hat{W} $$
Where $W$ is the original message (the mask in our simulation, a 32x32 array with 1s and
0s), $f_n$ is the encoder (the convolution with the Point Spread Function of the diffuse
light), $p(y|x)$ is the channel (the evolution through space and time of the diffused
light), $g_n$ is the decoder (the inverse-problem algoritm, which could for instance be
neural networks), and $\hat{W}$ is the reconstructed message. Here $X$ and $Y$ represent
the space of messages transmitted and received, respectively. The dimensions here are
important, since we don't have a usual time-series (which are the recurrent forms used in
digital signal processing), but rather $NxN$ time-series. In our simulations $X$ is
constant in time, but $Y$ is not.

As far as I can understand, the research that has been done thus far has been by
considering the channel to input and output in dimensions $NxN$ rather than $NxNxT$, by
integrating the received message over time or using other methods. We'd like to know the
mutual information with each approach, and see if considering the extra dimension we win
information.

There's yet another aspect I find wandwavy to transfer from IT to these physics concepts:
The channel capacity defines the maximum rate at which information can be transmitted
(rate is bits per unit time). But the model of diffuse light does not really transmit
information per unit time. Perhaps we could argue that the capacity here just tells the
maximum number of bits transmitted, period. Another way of seeing it is that it could map
to the number of bits lost in transmission by comparing it to the original amount of
information of the system.

## What is the current state of the field?
## What advances have been made?
