# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 16:18:05 2018

@author: Etien & Patr & Benjm
"""

def ezfft(t, S, normalization = "ortho", neg = False):
    """ Returns the Fourier transform of S and the frequency vector associated with it"""
    import numpy as np
    y = np.fft.fft(S,norm=normalization)
    y = np.fft.fftshift(y)
    f = np.fft.fftfreq(t.shape[-1], d = t[2]-t[1])
    f = np.fft.fftshift(f)
    if neg == False:
        y = 2*y[f>0]
        f = f[f>0]
    
    return f,y


def ezifft(f, y, normalization = "ortho"):
    '''Returns the inverse Fourier transform of y and the time vector associatedwith it
    WARNING : the negative frequencies must be included in the y vector'''
    import numpy as np
    N = len(f)
    tstep = 1/(N*(f[2]-f[1]))
    x = np.linspace(-(N*tstep/2),(N*tstep/2),N)
    y = np.fft.ifftshift(y)
    S = np.fft.ifft(y,norm=normalization)
    
    return x,S


## Ancienne fonction d'Etienne que j'ai modifié et je sais pas si ça fait une différence avec le shift
#def ezfft(t, S, normalization = "ortho"):
#    """ Returns the Fourier transform of S and the frequency vector associated with it"""
#    import numpy as np
#    y = np.fft.fft(S, norm = normalization)
#    f = np.fft.fftfreq(t.shape[-1], d = t[2]-t[1])
#    y = 2*y[f>0]
#    f = f[f>0]
#    
#    return f, y 
    



def ezsmooth(x,window_len=11,window='flat'):
     """smooth the data using a window with requested size.
     
     This method is based on the convolution of a scaled window with the signal.
     The signal is prepared by introducing reflected copies of the signal 
     (with the window size) in both ends so that transient parts are minimized
     in the begining and end part of the output signal.
     
     input:
         x: the input signal 
         window_len: the dimension of the smoothing window; should be an odd integer
         window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
             flat window will produce a moving average smoothing.
 
     output:
         the smoothed signal
         
     example:
 
     t=linspace(-2,2,0.1)
     x=sin(t)+randn(len(t))*0.1
     y=smooth(x)
     
     see also: 
     
     numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
     scipy.signal.lfilter
     """ 
     import numpy as np

     if x.ndim != 1:
         raise ValueError("smooth only accepts 1 dimension arrays.")
 
     if x.size < window_len:
         raise ValueError("Input vector needs to be bigger than window size.")
         
 
     if window_len<3:
         return x
     
     
     if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
         raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
     
 
     s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]

     if window == 'flat': #moving average
         w=np.ones(window_len,'d')
     else:
         w=eval('np.'+window+'(window_len)')
     
     y=np.convolve(w/w.sum(),s,mode='valid')
     return  y[round(window_len/2-1):-round(window_len/2)]   
     
     
def ezcorr(x, y1, y2, unbiased=False, Norm=False, Mean = False):
	
    '''Arguments : x, y1, y2, biased, norm
    This function assumes the x arrays are the same, "unbiased" = True for an unbiased calculation and "Norm" = False to not normalize
    One can not have "Norm" = True and "unbiased" = False'''
    import numpy as np
    if Mean is True:
        y1 = y1-y1.mean()
        y2 = y2-y2.mean()
        
    delta_t = x[1]-x[0]
    ord_corr = np.correlate(y1,y2,"same")*delta_t
    absc_corr = delta_t*np.linspace(-len(ord_corr)/2,len(ord_corr)/2,len(ord_corr))
    if unbiased == True:
        if Norm == True:
            ord_unbiased = np.empty(len(absc_corr))
            for k in range(len(ord_corr)):
                ord_unbiased[k] = ord_corr[k]/(len(ord_corr)-abs(k-int(len(ord_corr)/2)))
            return(absc_corr,ord_unbiased)
        else:
            ord_unbiased = np.empty(len(absc_corr))
            for k in range(len(ord_corr)):
                ord_unbiased[k] = ord_corr[k]/(len(ord_corr)-abs(k-int(len(ord_corr)/2)))/delta_t
            return(absc_corr,ord_unbiased)
    else:    	
        return(absc_corr,ord_corr)
        
        
def ezcsvload(filename, nbrcolumns = 2, delimiter = '\t', decimalcomma = False, outformat = 'array', skiprows = 0, profile = None):
    """ 
    Function for easy loading of csv-type files. Loading parameters can be set manually,
    or instruments-specific "profiles" can be called.
    
    """
    
    # Loading modules
    import csv
    import numpy as np

    # Load profile, if any is specified
    if profile is not None:
        if profile is 'HR2000':
            nbrcolumns = 2
            delimiter = '\t'
            decimalcomma = False
            skiprows = 14
            
        if profile is 'testfile':
            nbrcolumns = 3
            delimiter = ';'
            decimalcomma = False
            skiprows = 1
        if profile is 'OSA':
            nbrcolumns = 2
            delimiter = '\t'
            decimalcomma = True
            skiprows = 0
            
        # Add your own profiles here
            
    # Preallocate output list
    outlist = [ [] for var in range(nbrcolumns) ]
    
    # Load file
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=delimiter, quotechar='|')
        for index, row in enumerate(spamreader):
            
            if index >= skiprows: # Skip first "skiprows" rows of file
                
                for outlist_index in range(nbrcolumns):
                    
                    if decimalcomma is True: # Convert comma to marks
                        tmp = str(row[outlist_index])
                        tmp = tmp.replace(',','.')
                        outlist[outlist_index].append(float( tmp ))
                        
                    else:
                        outlist[outlist_index].append(float(row[outlist_index]))
    
    if outformat is 'array': # Convert lists of values to numpy arrays
        for outlist_index in range(nbrcolumns):
            outlist[outlist_index] = np.array(outlist[outlist_index])
    
    return outlist


def ezfindwidth(x, y, halfwidth = False, height = 0.5, interp_points = 1e6):
    import numpy as np
    
    
    # Max. / min. values of y array
    ymin = np.nanmin(y)
    ymax = np.nanmax(y - ymin)
    
    # Normalize
    ytmp = (y - ymin) / ymax
    
    # Interpolate data for better accuracy, if necessary
    if interp_points > len(x):
        # Interpolate data inside search domain for better accuracy
        xinterp = np.linspace(x[0] , x[-1] ,int(interp_points))
        yinterp = np.interp(xinterp,x,ytmp)
    else:
        xinterp = x
        yinterp = y
    
    # Cut interpolation domain at desired height
    tmp1 = (yinterp >= height )
    
    # Ensure there's a uniquely defined width 
    tmp2 = np.linspace(0,len(tmp1)-1,len(tmp1),dtype = int)
    tmp2 = tmp2[tmp1]
    tmp3 = tmp2[1:] - tmp2[:-1]
    
    # Output width, if defined
    if any(tmp3 > 1) | (len(tmp2)==0):
        width = np.nan
    else:
        width = xinterp[tmp2[-1]] - xinterp[tmp2[0]]
    
    # Divide by two, if desired
    if halfwidth is True:
        width /= 2

    return width



def ezdiff(x, y, n = 1, order = 2):
    """ Numerical differentiation based on centered finite-difference formulas.
    Outputs d^n/dx^n(y) and a appropriately truncated x vector. "order" parameter
    determines the order of the finite difference formula; it must be an even
    number greater than 0. High values of order can help precision or lead to 
    significant numerical errors, depending on the situation. x needs to increase
    monotonically in constant increments dx.
    """
    
    import numpy as np    # Fast numerical operations
    import math           # Used for floor and factorial

    # X increments
    dx = x[1] - x[0]

    # Number of finite difference coefficients to calculate
    nbr_coeff = 2 * math.floor( (n+1)/2 ) - 1 + order
        
    # p number (see wikipedia for more info)
    p = int((nbr_coeff - 1)/2)
        
    # Matrix of linear system Ax = b to solve
    Amatrix = np.zeros((nbr_coeff,nbr_coeff))
    for jj in range(nbr_coeff):
        tmp = -p + jj
        for kk in range(nbr_coeff):
            Amatrix[kk,jj] = tmp**kk
       
    # b vector
    bvector = np.zeros(nbr_coeff)
    bvector[n] = math.factorial(n)
      
    # Solve to find "x" vector, not related to "x" input
    # (sorry if this confusing)
    xvector = np.linalg.solve(Amatrix,bvector)
       
    # Preallocation
    deriv = np.zeros_like(x[p:-p])
    c = np.zeros_like(x[p:-p])
    
    # Evaluating finite difference, using Neumaier's improved Kahan summation
    # algorithm. Using it should reduce numerical errors during summation,
    # although it slows down calculations
    for ll in range(-p,p+1):
          
        new_term = (xvector[ll+p] * y[p+ll : len(y)-p+ll])/(dx**n)
        
        tmp1 = deriv + new_term
        
        
        cond = np.abs(deriv) >= np.abs(new_term)
        not_cond = np.logical_not(cond)
        
        c[cond] = c[cond] + (deriv[cond] - tmp1[cond]) + new_term[cond]
        
        c[(not_cond)] = c[(not_cond)] + (new_term[(not_cond)] - tmp1[(not_cond)]) + deriv[(not_cond)]
        
        
        deriv = tmp1
    
    # Requested derivative, corrected
    deriv += c
        
    # Appropriately truncated x vector for math and plotting
    xtrunc = x[p:-p]
    
    
    return xtrunc, deriv
