# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 16:18:05 2018

@author: Etien & Patr & Benjm

"""

"""
TEMPLATE FOR CREATION OF NEW FUNCTIONS, CLASSES, OR METHODS:
    
"""

#%% Imported modules
import numpy as np
import csv
import math
import scipy.constants as sc
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit


#%% Set constants and recurrant functions
C = sc.c                          # Speed of light
pi = sc.pi                        # Pi
from numpy import sqrt       # Square root
from numpy import log         # Natural logarithm
from numpy import exp         # Exponential

#%% Methods and classes

def ezfft(t, S, normalization = "ortho", neg = False):
    """ 
    Description: Returns the Fourier transform of S and the frequency vector associated with it
    
    Inputs:
        - t: array_like
            Time vector [seconds]
        - S: array_like
            Signal vector [arb.u.]
        - normalization: str, optional
            Type of normalization for fft algorithm. For more info, see https://docs.scipy.org/doc/numpy/reference/routines.fft.html#module-numpy.fft
        - neg: bool, optional
            Choose whether or not to include negative frequencies components in output.
    
    Outputs:
        - f: ndarray
            Frequency vector [Hz]
        - y: complex ndarray
            Complex amplitude vector [arb.u.]
    """
    y = np.fft.fft(S,norm=normalization)
    y = np.fft.fftshift(y)
    f = np.fft.fftfreq(t.shape[-1], d = t[2]-t[1])
    f = np.fft.fftshift(f)
    if neg == False:
        y = 2*y[f>0]
        f = f[f>0]

    return f,y


def ezifft(f, y, normalization = "ortho"):
    """
    Description: Returns the inverse Fourier transform of y and the time vector associated with it.
    
    Inputs:
        - f: array_like
            Frequency vector [Hz]
        - y: array_like
            Complex amplitude vector [arb. u.]
        - normalization: str, optional
            Type of normalization for fft algorithm. For more info, see https://docs.scipy.org/doc/numpy/reference/routines.fft.html#module-numpy.fft
    
    Outputs:
        - t: array_like
            Time vector [seconds]
        - S: array_like
            Signal vector [arb.u.] 
    
    Other comments: Negative frequencies components should be included in f and y vectors. Add them manually if needed. 
    """
    N = len(f)
    tstep = 1/(N*(f[2]-f[1]))
    x = np.linspace(-(N*tstep/2),(N*tstep/2),N)
    y = np.fft.ifftshift(y)
    S = np.fft.ifft(y,norm=normalization)

    return x,S




def ezsmooth(x, window_len=11, window='flat'):
     """
     Description: Smooth the data using a window with requested size.
         This method is based on the convolution of a scaled window with the signal.
         The signal is prepared by introducing reflected copies of the signal
         (with the window size) in both ends so that transient parts are minimized
         in the begining and end part of the output signal.

     Inputs:
         - x: array_like
             Input signal to smooth
         - window_len: int, optional
             The dimension of the smoothing window; should be an odd integer
         - window: str, optional
             The type of window used, can be any from 'flat', 'hanning', 'hamming', 'bartlett' or 'blackman'.
             "flat" window will produce a moving average smoothing.

     Outputs:
         - y: ndarray
             Smoothed signal
     """

     if x.ndim != 1:
         raise ValueError("smooth only accepts 1 dimension arrays.")

     if x.size < window_len:
         raise ValueError("Input vector needs to be bigger than window size.")


     if window_len<3:
         return x
     
     if (window_len % 2) == 0:
        window_len+=1


     if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
         raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


     s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

     if window == 'flat': #moving average
         w=np.ones(window_len,'d')
     else:
         w=eval('np.'+window+'(window_len)')

     y=np.convolve(w/w.sum(),s,mode='valid')
     return  y[int(np.ceil(window_len/2-1)):-int(np.ceil(window_len/2-1))]


def ezcorr(x, y1, y2, unbiased=False, Norm=False, Mean = False):

    '''
    Description: Returns the correlation between two vectors y1 and y2. Returns the autocorrelation if y1 = y2.
    
    Inputs: 
        - x: array_like
            time_like vector [arb.u.]
        - y1: array_like
            First signal vector [arb.u.]
        - y2: array_like
            Second signal vector [arb.u.]
        - Bias: bool
            Method for removing the bias
        - Norm: bool
            'Normalized' or 'coef' method 
        - Mean: bool
            Option to remove the mean of the signal before correlation
            
    This function assumes the x arrays are the same, "unbiased" = True for an unbiased calculation and "Norm" = False to not normalize
    One can not have "Norm" = True and "unbiased" = False. For more information on unbiasing types, see: https://www.mathworks.com/help/matlab/ref/xcorr.html
    '''
    
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
    Description: Function for easy loading of csv-type files. Loading parameters can be set manually,
        or instruments-specific "profiles" can be called. Returns a list of array_like data.
    
    Inputs:
        - filename: str
            Name of the file to load. If not in the same folder, include path.
        - nbrcolumns: int, optional
            Number of columns contained in the file
        - delimiter: str, optional
            Character separating elements of different columns
        - decimalcomma: bool, optional
            Indicate whether or not the file use a comma to identify decimals
        - outformat: str, optional
            If "array", convert lists containing data from each columns into numpy arrays
        - skiprows: int, optional
            If file contains a header, indicate how many rows it contains to skip them.
        - profile: str, optional
            Automatically set all other options to match file format of a specific instrument.
            
    Outputs:
        - outlist: list
            Data extracted from file. Each element in the list represents a column from the file.

    """

    # Load profile, if any is specified
    if profile is not None:
        if profile is 'HR2000':
            nbrcolumns = 2
            delimiter = '\t'
            decimalcomma = False
            skiprows = 0

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
                        try:
                            outlist[outlist_index].append(float( tmp ))
                        except:
                            print("1 data point failed")

                    else:
                        try:
                            outlist[outlist_index].append(float(row[outlist_index]))
                        except:
                            print("1 data point failed")

    if outformat is 'array': # Convert lists of values to numpy arrays
        for outlist_index in range(nbrcolumns):
            outlist[outlist_index] = np.array(outlist[outlist_index])

    return outlist


def ezfindwidth(x, y, halfwidth = False, height = 0.5, interp_points = 1e6, pos = False):
    """
    Description: Function that finds the width of an input signal or pulse. By default, the FWHM will be returned, unless height 
                 and/or halfwidth options are changed. 
                 
    Inputs: 
        - x: array_like 
            Abscissa. Width will be defined in the same units (time, frequency, wavelength, etc.) 
        - y: array_like 
            Signal/pulse to measure [arb. u.]
        - halfwidth: bool, optional
            If True, will return the halfwidth
        - height: float, optional
            Selects height for which width is calculated
        - interp points: int, optionnal
            Function will be interpolated to this number of points if the input has less points than this number
        - pos: bool, optional
            if true return the position array 
        
    Outputs:
        - width (defined in the same units (time, frequency, wavelength, etc.))
        - xvalues: array, optional
            The two positions in x used for the width calculation
    """
    # Ensure x is stricktly ascending
    IIsort = np.argsort(x)
    x = x[IIsort]
    y = y[IIsort]
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
        
    if pos is True:
        xvalues = np.array([tmp2[-1], tmp2[0]])
        return width, xvalues
    else:
        return width



def ezdiff(x, y, n = 1, order = 2):
    """ 
    Description: Numerical differentiation based on centered finite-difference formulas.
    
    Inputs: 
        - x: array_like; increases monotically in constant increments dx
            abscissa
        - y: array_like
            ordonate
        - n: int
            diffirentiates 'n' times
        - order: 
            "order" parameter determines the order of the finite difference formula; 
            it must be an even number greater than 0. High values of order can help 
            precision or lead to significant numerical errors, depending on the situation.
            2 is usually a safe value.
   Outputs:
       - xtrunc: array_like
           truncated abscissa
       - deriv: array_like
           nth derivative of y with respect to x.
    """

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

def knife_edge_experiment(z = None, P = None, P0 = 0, P_max = None, plot = True):
    
    """
    Description: With the measured parameters from a knife edge experiment, finds the beam size
                 by fitting a error function on the data. The function on which the data is fitted is the following:
                 P0 + 0.5*P_max*( 1 - erf(sqrt(2)*(z -z0)/w) )
                 z0 and w are the fitting parameters. P_max is also a fitting parameter if not specified as input.
    
    Inputs: 
        - z: array_like
            vector containing the positions of the stage [mm]
        - P: array_like
            Singal (power) vector corresponding to the stage positions [mW]
        - P0: float, optionnal
            Signal (power) in absence of the incident beam
        - P_max: float, optional
            Maximum signal (power). i.e. when the beam is not blocked. 
            If P_max is not entered, the function will use it as another
            fitting parameter and display the computed value.
        - plot: bool, optionnal
            Plots the measured and fitted Power curves if True
    Outputs:
        - params: array
            contains in that order: z0, w, P_max (if not specified as input)
    """
    
    if z is None or P is None:
        print('Position or power array is missing')
        return
    
    if type(z) != np.ndarray or type(P) != np.ndarray:
        print('z and P must be numpy arrays')
        return
    
    if len(P) != len(z):
        print('z and P must be the same length')
        return
    
    # Defining error function
    if P_max:
        def func(z, z0, w):
            return P0 + 0.5*P_max*(1-sp.special.erf(np.sqrt(2) * (z -z0)/w))
        # Fit an error function on the data
        params, param_covar = curve_fit(func, z, P)
        z_fit = np.linspace(0.8*np.min(z), 1.2*np.max(z), 100)
        P_fit = func(z_fit, params[0], params[1])
    else:
        def func(z, z0, w, P_max):
            return P0 + 0.5*P_max*(1-sp.special.erf(np.sqrt(2) * (z -z0)/w))
        # Fit an error function on the data
        params, param_covar = curve_fit(func, z, P, bounds=(0, [np.max(z), np.max(z), np.max(P)]))
        z_fit = np.linspace(0.8*np.min(z), 1.2*np.max(z), 100)
        P_fit = func(z_fit, params[0], params[1], params[2])
        print('The fitted max power is: ' + str(params[2]))
    
    print('The beam diameter (1/e^2) is: ' + str(2*params[1]) + ' mm')
    
    # Plotting
    if plot is True:
        import femtoQ.plotting as fqp
        fqp.set_default_values_presentation()
        plt.figure()
        plt.plot(z,P, 'o', label = 'Measured')
        plt.plot( z_fit, P_fit, label = 'Fitted')
        plt.xlabel('Razor edge position (mm)')
        plt.ylabel('Average power (mW)')
        if P_max:
            plt.ylim([0,1.05*P_max])
        else:
            plt.ylim([0,1.05*params[2]])
        plt.legend()
    
        plt.show()
    
    # Returns
    return params
    
def ezpad(x, y, left, right, values=(50,50)):
    """
    Input :
    - x is the abscissa to pad, it is assumed to be a linear list
    - y is the ordinate that will be padded
    - left is the number of values to add to the left
    - right is the number of values to add to the right
    - values is (value to add to the left, value to add to the right)
    Output :
    Padded x and y lists
    """
    y = np.pad(y, (left, right), 'constant', constant_values=values) # Use numpy pad function to pad y array
    step = x[1] - x[0]
    # Manually padding the x vector assuming it is linear
    x = np.append(np.linspace(x[0]-left*step, x[0]-step, left), x)
    x = np.append(x, np.linspace(x[-1]+step, x[-1]+right*step, right))
    return x,y

class Pulse:
    """
    Description: Class to model simple electromagnetic pulse propagation and interactions.
        The pulse is modled by its electric field in the time domain.
        
    Input:
        ----------------------------------------------------------
        - lambda0: float
            Center wavelength of pulse [meters]
        - tau_FWHM: float
            FWHM pulse duration [seconds]    
        OR
        
        - v0: float
            Center frequency of pulse [Hertz]
        - v_bandwidth: float
            FWHM bandwidth in frequency domain [Hertz]
        OR
          
        - E: ndarray
            Electric field-like vector, defined in time domain [arb. u.]
        - t: ndarray
            Time vector [seconds]
        -----------------------------------------------------------
        &
        
        - A: float, optional
            Maximum amplitude of E field, when E is not inputed [arb. u.]
        - T: float, optional
            Time window represented by time vector, when t is not inputed [seconds]
        - dt: float, optional
            Resolution of the time vector, when t is not inputed [seconds]
    
    Attributes:
        - E: ndarray
            Electric field-like vector, defined in time domain. Is either inputed or calculated.
        - t: ndarray
            Time vector. Is either inputed or defined as a linear grid of resolution dt and length T
    
    Methods:
        - disperse: Returns a Pulse object on which effects of dispersion have been applied, either "manually" or through linear propagation modeling
        
        - getFWHM: Returns FWHM of either: pulse's intensity in time, power spectrum in frequency domain or power spectrum in wavelength domain
        
        - getInstFreq: Returns instantaneous angular frequency as a function of time
        
        - SFG: Returns a Pulse object calculated as a perfectly phase-matched SFG interaction between two Pulse objects.
                Allows to estimate SFG spectrum relative shape.
      
        - delay: Returns a Pulse object which has been delayed in time by a fixed amount.
    """
    def __init__(self, lambda0 = None, tau_FWHM = None, v0 = None, v_bandwidth = None, A = 1, t = None, E = None, T = 10000e-15, dt = 0.1e-15):
        #%% Simulation parameters
        self.T = T
        self.dt = dt
        if ((t is not None) & (E is not None)):
            self.t = t
            self.E = E

        elif ((lambda0 is not None) & (tau_FWHM is not None)):

            # Other pulse parameters
            tau = tau_FWHM / sqrt(2*log(2)) # Gaussian pulse half width @ e^-2
            v0 = C/lambda0                  # Pulse's carrier frequency
            w0 = 2*pi*v0                    # Pulse's carrier angular frequency

            # Electric field in time domain
            self.t = np.linspace(-T/2,T/2, round(T/self.dt) )
            self.E = A*np.exp(1j * w0 * self.t) * np.exp(-  (self.t)**2 / (tau)**2)

        elif ((v0 is not None) & (v_bandwidth is not None)):
            tauFWHM = 0.44/v_bandwidth         # Initial pulse duration, full width at half maximum
            tau = tauFWHM / sqrt(2*log(2)) # Gaussian pulse half width @ e^-2
            w0 = 2*pi*v0                    # Pulse's carrier angular frequency
            self.t = np.linspace(-T/2,T/2, round(T/self.dt) )
            self.E = A* exp(1j * w0 * (t)) * np.exp( - (t)**2 / (tau)**2)    # Electric field of the pulse in the time domain

    def disperse(self, medium = None, L = None, dispVec = None, v0 = "Auto"):
        """
        Description: This method takes the pulse and retrieves the temporal shape of the electric field
            after propagation in a given medium, using Sellmeier's equations of this medium. If an
            optionnal manual_GGD is entered, the code will not use "medium" and "L".

        Inputs:
            - material: str
                The optical medium through which the pulse propagates
            - L: float
                The thickness of material through which the pulse propagates [meters]
            - dispVec: array-like, optional
                Custom amount of dispersion. Vector containing orders of dispersion from 2 up to whatever is needed. 
                Example input is [GDD, TOD, ...], with GDD in fs^2, TOD in fs^3 and so forth.
            - v0: float or "Auto"
                Central frequency [Hertz]. If "Auto", will be calculated as average frequency of power spectrum.
        
        Outputs:
            - new_Pulse: Pulse object
                Dispersed pulse
        """
        # v is freqnecy vector, s is frequency-domain electric field vector
        v,s = ezfft(self.t,self.E, neg = True)
        # Calculate mean frequency
        if dispVec is not None:
            if v0 == "Auto":
                v0 = np.trapz(np.abs(s)**2 * v)/np.trapz(np.abs(s)**2)
            # conversion in angular frequency
            w0 = 2*pi*v0
            w = 2*pi*v
            # Calculate all phase orders
            phase = np.zeros_like(v)
            for i, disp in enumerate(dispVec):
                phase += 1/math.factorial(i+2)*disp*(w-w0)**(i+2)*(1e-15)**(i+2)
            s = s*exp(1j*phase)
        else:
            if medium.lower() == "bk7":
                lambda_1 = 0.3e-6   # Cutoff wavelengths of the equation's validity
                lambda_2 = 2.5e-6
                n_sellmeier = lambda x: (1+1.03961212/(1-0.00600069867/x**2)+0.231792344/(1-0.0200179144/x**2)+1.01046945/(1-103.560653/x**2))**.5  #https://refractiveindex.info/?shelf=glass&book=BK7&page=SCHOTT
            elif ((medium.lower() == "fs") or (medium.lower() =="fused silica")):
                lambda_1 = 0.21e-6
                lambda_2 = 6.7e-6
                n_sellmeier = lambda x: (1+0.6961663/(1-(0.0684043/x)**2)+0.4079426/(1-(0.1162414/x)**2)+0.8974794/(1-(9.896161/x)**2))**.5       #https://refractiveindex.info/?shelf=glass&book=fused_silica&page=Malitson
            elif medium.lower() == "yag":
                lambda_1 = 0.4e-6
                lambda_2 = 5e-6
                n_sellmeier = lambda x: (1+2.28200/(1-0.01185/x**2)+3.27644/(1-282.734/x**2))**.5   #https://refractiveindex.info/?shelf=main&book=Y3Al5O12&page=Zelmon
            elif medium.lower() == "sf11":
                lambda_1 = 0.37e-6
                lambda_2 = 2.5e-6
                n_sellmeier = lambda x: (1+1.73759695/(1-0.013188707/x**2)+0.313747346/(1-0.0623068142/x**2)+1.89878101/(1-155.23629/x**2))**.5      #https://refractiveindex.info/tmp/data/glass/schott/N-SF11.html
            elif medium.lower() == "sf10":
                lambda_1 = 0.38e-6
                lambda_2 = 2.5e-6
                n_sellmeier = lambda x: (1+1.62153902/(1-0.0122241457/x**2)+0.256287842/(1-0.0595736775/x**2)+1.64447552/(1-147.468793/x**2))**.5    #https://refractiveindex.info/?shelf=glass&book=SF10&page=SCHOTT
            elif medium.lower() == "znse":
                lambda_1 = 0.54e-6
                lambda_2 = 18.2e-6
                n_sellmeier = lambda x: (1+4.45813734/(1-(0.200859853/x)**2)+0.467216334/(1-(0.391371166/x)**2)+2.89566290/(1-(47.1362108/x)**2))**.5    #https://refractiveindex.info/?shelf=main&book=ZnSe&page=Connolly
            else:
                print("The entered medium does not exist or its Sellmeier's equations are not contained in this method")
                return self

            #%% Dispersive propagation
            # Convert Sellmeier's equations cutoff wavelengths to frequencies
            v1 = C / lambda_2
            v2 = C / lambda_1

            # Convert frequencies within the equation's validity domain to wavelengths in um
            vtmp = v[( (v>v1) & (v<v2) )]
            lambdatmp = C / vtmp

            # Calculate refractive index for relevant frequencies
            ntmp = n_sellmeier(lambdatmp*1e6)

            # Generate refractive index vector n(v). Values outside of validity range are
            # set to zero.
            n = np.ones_like(v)
            n[( (v>v1) & (v<v2) )] = ntmp

            # Apply material's spectral phase to frequency domain electric field
            s[v!=0] = s[v!=0] * np.exp(1j * 2 * pi * n[v!=0] * (v[v!=0] / C) * L)

        #%% Inverse fast fourier transform
        # E2 is final time domain electric field vector, t2 is its corresponding time
        # vector. If all works as intended, t2 is equivalent to t at this point
        t2, E2 = ezifft(v,s)

        #%% Final pulse realignment
        # Find final pulse's peak in time
        tpeak = np.mean( t2[np.abs(E2)**2 == np.nanmax(np.abs(E2)**2)] )

        tExtended = np.concatenate((t2-np.abs(np.min(t2)) - np.max(t2) - self.dt, t2, t2+np.abs(np.min(t2)) + np.max(t2) + self.dt), axis = 0 )

        EExtended = np.pad(E2, E2.shape, mode = 'wrap')

        # Shift time vector
        E2 = np.interp(t2, tExtended-tpeak, EExtended)

        # New object
        new_Pulse = Pulse(t = t2,E = E2)
        return new_Pulse

    def getFWHM(self, domain = "wavelength"):
        """
        Description: Returns the FWHM of the pulse, defined in either of these 3 domains, as selected by the user:
            - wavelength (default)
            - frequency
            - time
            
        Inputs:
            - domain: str, optional
                Domain in which to evaluate FWHM
                
        Outputs:
            - FWHM: float
        """
        v, s = ezfft(self.t, self.E)
        if domain is "time":
            return ezfindwidth(self.t, np.abs(self.E)**2)
        elif domain is "frequency":
            return ezfindwidth(v, np.abs(s)**2)
        elif domain is "wavelength":
            l = C/v
            return ezfindwidth(l, np.abs(s)**2)
        else:
            print("Invalid domain")
            return None


    def getInstFreq(self):
        """
        Description:  Returns instantaneous angular frequency as a function of time.
        
        No input whatsoever.
        
        Outputs:
            - w_inst: ndarray
                Instantanious angular frequency
        """
        # Temporal phase
        phi_t = np.unwrap(np.angle(self.E))
        # Instantaneous angular frequency
        w_inst = (phi_t[2:] - phi_t[:-2]) / (self.t[2] - self.t[0])
        # Add zeros the the extremities to keep the same vector length
        w_inst = np.insert(w_inst, 0, 0)
        w_inst = np.append(w_inst, 0)
        # Return instentanious frequency
        return w_inst

    def SFG(self, Pulse2, SFGonly = True):
        """
        Description: Returns the spectral amplitude of the SFG from Pulse1 and Pulse2, along with a new Pulse object
            defined by the total field resulting of the interaction. If SFGonly is False, the two SHGs will also
            be returned along with the SFG in both the field and the spectral amplitude.
            
        Inputs:
            - Pulse2: Pulse object.
                Other Pulse2 involved in SFG process. Input self for SHG.
            - SFGonly: bool
                If false, self-SHG of each input pulse will also be computed and be added to SFG field.
                
        Outputs:
            - SFG pulse: Pulse object
            - f_nl: ndarray
                Frequency vector for SFG spectrum [Hertz]
            - A_nl: ndarray
                Complex amplitude spectrum [arb. u.]
        """
        # Verification that the time vectors are the same
        if self.t.all() == Pulse2.t.all():
            # Consider SFG without SHGs
            if SFGonly is True:
                Atmp = self.E * Pulse2.E
            # Consider both SHGs and the SFG
            elif SFGonly is False:
                E_tot =  self.E + Pulse2.E
                Atmp = E_tot * E_tot

            f_nl, A_nl = ezfft(self.t, Atmp)
            return Pulse(t = self.t, E = Atmp), f_nl, A_nl
        else:
            print("time vector must be identical")
            return None


    def delay(self, timeDelay):
        """
        Description: Delay the pulse in time domain by a value equal to timeDelay. Does not wrap it back once it reaches the
            end of time vector t. A positive delay means the pulse's peak will shift towards negative time values.
            
        Inputs:
            - timeDelay: Time duration with which to delay the pulse [seconds]
            
        Outputs:
            - newPulse: Pulse object
                Delayed pulse.
        """
        
        newEfield = np.interp(self.t, self.t - timeDelay, self.E)
        
        newPulse = Pulse(t = self.t, E = newEfield)
        
        return newPulse










