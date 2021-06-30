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
## What is information theory?
## What is the goal?
The goal is to determine from an information theory point of view whether gathering extra
information about the scattering of light would help reconstruct the scattered object.
## What is the current state of the field?
## What advances have been made?
