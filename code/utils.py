"""
Created on Tue Aug 16 2021
@author: Allan Perez, School of Physics and Astronomy, University of Glasgow,
2480326p@student.gla.ac.uk

Here we will write helper functions. As of now, the following will be included:
- Information content of a message: 1D array input (x) and probability distribution (A_x), scalar output (bits of information)
- Joint entropy H(X,Y)
- Mutual information between two messages: Two 1D arrays input, scalar output (bits of
  mutual information).
- Parameters generator: Returns a dictionary with the required parameters to perform
  simulations. The parameters are editable, since we're using a dictionary.

Following the law of large numbers and its frequentist implications, these computations
will be most accurate when the messages are large at size. This is because the
probability distribution is estimated using a frequentist approach.
"""
import numpy as np

def generate_parameters(**kwargs):
    """args:
        kwargs: optional parameters of the simulation. In order to change a parameter, it
        must have the exact same name, otherwise it's ignored. The get method used below
        is necessary to handle the case when the given key doesn't exist.
    returns:
        Dictionary of parameters used for simulation.
    """
    n = kwargs.get('n') or 1.44                   # refractive index of diffuse medium
    xypadding = kwargs.get('nBinsPadded') or 65 # Pixel count after 0 padding for the FFTs - avoids wrapping
    params={
        "c" : 3e10/n,             # phase velocity in medium (cm/s) -- Not optional
        "propagationLen1" : kwargs.get('propagationLen1') or 2.5,  # thickness of first diffuse medium (cm)
        "propagationLen2" : kwargs.get('propagationLen2') or 2.5,  # thickness of second diffuse medium (cm)
        "mu_a" : kwargs.get('mu_a') or 0.09,     # absorption coefficient (cm^-1)
        "mu_s" : kwargs.get('mu_s') or 16.5,     # scattering coefficient (cm^-1)
        "FoV_length" : kwargs.get('FoV_length') or 4.4,           # size of the camera field of view (cm)
        "timeResolution" :  kwargs.get('timeResolution') or 55e-12,    # camera temporal bin width (ps)
        "timeNumBins" : kwargs.get('timeNumBins') or 251,          # number of temporal bins
        "fieldOfViewBins" : kwargs.get('fieldOfViewBins') or 33,       # number of camera pixels
    }
    params["D"] = (3*(params["mu_a"]+params["mu_s"]))**(-1) # "D" as defined in paper

    # array of positionsin FoV (cm)
    _ = np.linspace(-params["FoV_length"]/2,params["FoV_length"]/2,params["fieldOfViewBins"])
    params["xCoordSpace"] = _
    params["yCoordSpace"] = _
    params["padSize"]= (xypadding - params["fieldOfViewBins"])//2
    return params

# The below methods, although functional, are useless because it can be more generally
# handled by sklearn's information content functions.

def check_message(message):
    if type(message) != np.ndarray:
        if type(message)==str:
            message = np.array(list(message))
        else:
            raise Exception("Message must be a string or np array.")
    return message

def information_content(message):
    """args:
        message: An array or string of N symbols. PDF is estimated from message.
    returns:
        Shannon information of message in bits and its upper bound
    Notes:
        For X=(x,A_x,P_x), A_x is the set of symbols used in message, and P_x is its
        distribution function frequentially estimated from message. Then H(X) is the
        average information content per symbol. So the information content of the message
        is the average information content per symbol times the number of symbols used in
        the message, since it's modelled as an element of the joint ensamble XXXX...X, and
        by taking it to be independent, the information content is additive, hence the
        multiplication.
    """
    message = check_message(message)

    A_x, F_x = np.unique(message, return_counts=True)
    sizem = message.size
    P_x = [ fx/sizem for fx in F_x ]
    H = sum([px *np.log2(1/px) for px in P_x])
    H_max = np.log2(A_x.size)
    informationContent = H*sizem
    informationContentMax = H_max*sizem
    return (informationContent, informationContentMax)

def conditional_information_content(message, prior):
    """args:
        message: An array or string of N symbols. PDF is estimated from message.
        prior: An array or string of N symbols. PDF is estimated from message.
    returns:
        Conditional shannon information of the message given prior, in bits.
    Notes:
        H(X|Y)= sum_xy P(x,y) log(1/P(x|y))
    """
    mX = check_message(message)
    mY = check_message(prior)
    if mX.size != mY.size:
        raise Exception("Conditional entropy is defined only for message and priors of\
                        same size")

    A_x, F_x = np.unique(mX, return_counts=True)
    A_y, F_y = np.unique(mY, return_counts=True)
    mXmY = np.transpose([mY,mX])

    A_xA_y, F_xy = np.unique(mXmY, return_counts=True, axis=0)
    sizem = mX.size

    P_xy = [ fxy/sizem for fxy in F_xy] #P(x,y)
    P_y  = [ fy/sizem for fy in F_y] # P(y)
    P_y_d = dict([ [a_y, p_y] for a_y, p_y in zip(A_y, P_y)])

    ylabels = A_xA_y[:,0]
    P_x_y= [ p_xy/P_y_d[y] for p_xy,y in zip(P_xy, ylabels)] #P(x|y) = P(x,y)/P(y)
    H_X_Y = sum([ p_xy*np.log2(1/p_x_y)for p_xy,p_x_y in zip(P_xy, P_x_y)])
    conditional_information_content = H_X_Y * sizem
    return conditional_information_content

def mutual_information(m_X, m_Y):
    """args:
        m_X: An array or string of N symbols. PDF is estimated from message.
        m_Y: An array or string of N symbols. PDF is estimated from message.
    return:
        I(X;Y) estimated from the message's symbols distribution.
    """
    return information_content(m_X)[0] - conditional_information_content(m_X,m_Y)

if __name__=='__main__':
    m = '0123456789'
    #m = '1782345678946789168976789239678231678912346789'
    m0 = ['3113331', '010011101', '101101', '101101100', '0123456789', '0123456789']
    m1 = ['4347374', '433343444', '202232', '101101100', '9876543210', 'asdfghjklz']
    HX = [information_content(m) for m in m0]
    HY = [information_content(m) for m in m1]
    I0 = [mutual_information(mx,my) for mx,my in zip(m0,m1)]
    I1 = [mutual_information(my,mx) for mx,my in zip(m0,m1)]
    print("Information content 0: ", *HX)
    print("Information content 1: ", *HY)
    print("Mutual info 0: ", *I0)
    print("Mutual info 1: ", *I1)
