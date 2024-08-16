import numpy as np
from numpy.core.numeric import convolve
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from numba import jit
from scipy import optimize as opt
from astropy.convolution import convolve_fft
from astropy.stats import sigma_clip
from irap_tools import paths

@jit(nopython=True)
def polynom(x,coeffs):#, degree=3):
    """
    Returns value for a polynomial up to degree 5. The degree is dertermined
    by the lengths of coeffs.
    The value returns correspond to a polynom of the type:
    ..math: coeffs[0]*x**5+coeffs[1]*x**4+coeffs[2]*x**3+
            coeffs[3]*x**2+coeffs[4]*x+coeffs[5]

    The first element of coeffs is always the coefficient of the highest 
    degree, and the last of element is always the coefficient of degree 0.

    Input parameters:
    - x         :   for which to compute the output
    - coeffs    :   polynomial coefficients

    Output parameters:
    - _         :   Computed value(s) for the len(coeffs)-1 degree polynomial.
    """

    degree = len(coeffs)-1
    # The order is important to have the correct combinations of permutations
    if degree==0:
        return 0*x + coeffs[0]
    elif degree==1:
        return coeffs[0]*x+coeffs[1]
    elif degree==2:
        return coeffs[0]*x**2+coeffs[1]*x + coeffs[2]
    elif degree==3:
        return coeffs[0]*x**3+coeffs[1]*x**2+coeffs[2]*x + coeffs[3]
    elif degree==4:
        return coeffs[0]*x**4+coeffs[1]*x**3+coeffs[2]*x**2+\
            coeffs[3]*x+coeffs[4]
    elif degree==5:
        return coeffs[0]*x**5+coeffs[1]*x**4+coeffs[2]*x**3+\
            coeffs[3]*x**2+coeffs[4]*x+coeffs[5]
    else :
        # return np.array([0.])
        return 0*x
        print("degree not supported")

@jit(nopython=True)
def normalize_axis(a,b):
    '''
    method that modify array based on second array
    returns (a-np.mean(b))/(np.max(a)-np.min(b))    
    '''
    c = (a-np.mean(b))/(np.max(b)-np.min(b))
    return c

@jit(nopython=True)
def revert_normalize_axis(a,b):
    '''
    method that modify array based on second array
    returns a*(np.max(b)-np.min(b))+np.mean(b)   
    '''
    c = a*(np.max(b)-np.min(b))+np.mean(b)
    return c

@jit(nopython=True, cache=True)
def moving_median(Im,hws,btd=None, p=50):    
    """
    Compute the moving median of Im and divide Im by the resulting moving median computing within
    [W[N_bor],W[-N_bor]] with N_best points
    
    Inputs:
    - Im    :       1D exposure to normalize
    - hws   :       Half Window Size of the window used to compute the median
    - btd   :       Bins to delete on the edges [default is hws].
    
    Outputs:
    - I_nor: Normalised exposure (size: len(I_tm))
    - I_bin: Resulting moving median used to normalize the Im 
    """

    if btd is None:
        btd = hws
    ## Init the moving median
    # W_bin = np.empty(len(Wm)-N_bor)
    I_bin = np.empty(len(Im)) * np.nan
    
    ## Apply moving median
    for k in range(len(Im)):  
        ## Handle edges
        if k < hws:
            N_inf = 0
            N_sup = int(k+hws)
        elif k + hws > len(Im):
            N_inf = int(k-hws)
            N_sup = -1
        else:
            N_inf = int(k-hws)
            N_sup = int(k+hws)       
        # W_bin[k] = np.nanmedian(Wm[N_inf:N_sup])
        r = Im[N_inf:N_sup]
        # idx = sigma_clip(r,5,5) #Apply sigma-clipping
        # r = np.delete(r, idx)
        # if np.all(np.isnan(r)):
        #     I_bin[k] = np.nan    
        # else:
        # idx = ~np.isnan(r)
        I_bin[k] = np.nanpercentile(r, p)  #Take median

    ## Set borders to NaNs    
    I_bin[:btd] = np.nan
    I_bin[len(I_bin)-btd:] = np.nan
    I_nor = Im/I_bin

    return I_nor,I_bin


@jit(nopython=True)
def fit_1d_polynomial(x, z, degree=3, returnCovMat=False):
    '''
    !!!! Caution Feb 15, 2024; reliazed problems starting with degree 3... Is this equation relly correct?
    The function fit_poly relying on curve fit works better !
    
    Analytical determination of the parabolic parameters.
    Two methods are implemeted:
    - method 2 based on Louise Yu (2019)

    Input parameters:
    - x         :   Must be a 1D array
    - z         :   Must be a 1D array
    - degree    :   degree used to fit polynomial
    Output parameters:
    - o  :  1D array containing the values of the ten coefficients 
            a,d,f,b,c,e,g,h,i,j
    '''

    # Reshaping the arrays to 1D
    x = np.reshape(x, x.size)
    z = np.reshape(z, z.size)
    idx = np.where(np.isnan(z))
    x = np.delete(x, idx[0])
    z = np.delete(z, idx[0])
    ones = np.ones(x.shape)
    if degree==1:
        # Based on Louise Yu (2019)
        # Computing paraboloid coefficients
        A = np.stack((x, ones))
        A = np.dot(A,A.T)/len(x)
        B = np.stack((x, ones))*z
        Bsum = np.sum(B,1)/len(x)
        invA = np.linalg.inv(A)
        output = np.dot(invA, Bsum)
        # output = np.array([1,1])
        return output
    elif degree==2:
        ## Based on Louise Yu (2019)
        # Computing paraboloid coefficients
        A = np.stack((x**2, x, ones))
        A = np.dot(A,A.T)/len(x) ## Makes the mean of all components
        B = np.stack((x**2, x, ones))*z
        Bsum = np.sum(B,1)/len(x)
        output = np.dot(np.linalg.inv(A), Bsum)
        return output
    elif degree==3:
        ## Based on Louise Yu (2019)
        # Computing paraboloid coefficients
        A = np.stack((x**3, x**2, x, ones))
        A = np.dot(A,A.T)/len(x)
        B = np.stack((x**3, x**2, x, ones))*z
        Bsum = np.sum(B,1)/len(x)
        output = np.dot(np.linalg.inv(A), Bsum)
        return output
    elif degree==4:
        ## Based on Louise Yu (2019)
        # Computing paraboloid coefficients
        A = np.stack((x**4, x**3, x**2, x, ones))
        A = np.dot(A,A.T)/len(x)
        B = np.stack((x**4, x**3, x**2, x, ones))*z
        Bsum = np.sum(B,1)/len(x)
        output = np.dot(np.linalg.inv(A), Bsum)
        return output
    elif degree==5:
        ## Based on Louise Yu (2019)
        # Computing paraboloid coefficients
        A = np.stack((x**5, x**4, x**3, x**2, x, ones))
        A = np.dot(A,A.T)/len(x)
        B = np.stack((x**5, x**4, x**3, x**2, x, ones))*z
        Bsum = np.sum(B,1)/len(x)
        output = np.dot(np.linalg.inv(A), Bsum)
        return output
    elif degree==0:
        ## Based on Louise Yu (2019)
        # Computing paraboloid coefficients
        # A = np.array([np.ones(np.shape(x))])
        # A = np.dot(A,A.T)/len(x)
        # B = np.array([np.ones(len(x))])*z
        # B = np.sum(B,1)/len(x)
        # output = np.dot(np.linalg.inv(A), B)
        return np.array([np.median(z)])
    else:
        return np.array([0.])
        print("Degree not supported")
