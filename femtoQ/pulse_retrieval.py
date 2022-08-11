

#%% Import modules
import femtoQ.tools as fq
import numpy as np
import matplotlib.pyplot as plt
from femtoQ.pr_backend import tdsilib as library_2dsi
from femtoQ.pr_backend import froglib as library_frog
from femtoQ.pr_backend import dscanlib as library_dscan
from scipy.constants import c as C
from scipy.interpolate import interp1d as interp
from scipy.integrate import simps as simpson
import pypret

def freq2time(frequency, amplitude_spectrum):
    # Interpolate over new frequency grid, including negative w components
    freq_max = frequency[-1]
    freq_min = -freq_max
    N = len(amplitude_spectrum)
    dv = freq_max/N
    
    new_freq_grid = np.hstack((np.array([0]),np.linspace(dv,freq_max,N),np.linspace(freq_min,-dv,N)))
    
    spectrum_interpolated = interp(frequency,amplitude_spectrum,'quadratic',bounds_error=False,fill_value=0)(new_freq_grid)
     

    amplitude_time = np.fft.fftshift(np.fft.ifft(spectrum_interpolated))
    t = np.fft.fftshift(np.fft.fftfreq(2*N+1,dv))

    return t, amplitude_time


def twodsi(filename,  upconvWavelength = 'auto', wavelengthCutoffs = None, smoothSpectrum = False, zeroPadTrace = True, windowTrace = True, minTimeResolution = 1e-15, folder = '', spectrum = None,relativeNoiseFloor = 0.01,simulatedData = False,debug = False):
    
    
    data = np.load(folder+filename)
    fixedMirrorData = data['upconvSpectrum']
    wavelengths = data['wavelengths']
    shearStagePosition = data['shearStagePosition']
    movingMirrorData = data['shearTrace']
    twoDSIStagePosition = data['twoDSIStagePosition']
    tdsiTrace = data['twoDSITrace']
    
    #%% Load data
    if simulatedData:
        #tdsiTrace = data['twoDSITrace']
        timeVector = data['timeVector']
        inputPulse = data['inputPulse']
    
    data.close()
    
    
    #%% Cut spectrum to relevant spectral region
    """ Min and max wavelengths of spectral windowing. Any data outisde of this range
        will be discarded. """
        
    if wavelengthCutoffs is not None:
        minWavelength = wavelengthCutoffs[0]
        maxWavelength = wavelengthCutoffs[1]
        
        fixedMirrorData, movingMirrorData, wavelengths, tdsiTrace = library_2dsi.cut_spectrum(minWavelength, maxWavelength, fixedMirrorData, movingMirrorData, wavelengths, tdsiTrace)
    
    
    #%% Smooth spectrum, if desired
    if smoothSpectrum:
        fixedMirrorData = fq.ezsmooth(fixedMirrorData, window_len = 15, window = 'hanning')
        for ii in range(movingMirrorData.shape[0]):
            movingMirrorData[ii,:] = fq.ezsmooth(movingMirrorData[ii,:], window_len = 15, window = 'hanning')
    
    
    #%% Substract noise floor from data. Taken as max value of either the first or
    #   the last datapoint of the spectrum. Any negative value is afterward set to 0.
    fixedMirrorData, movingMirrorData = library_2dsi.substractNoiseFloor(fixedMirrorData, movingMirrorData,relativeNoiseFloor)
    
    """ Add trace postprocessing here. Mostly consists of interpolating trace to a linear
        grid along stage displacement axis. Will be added once the 2dsi control software
        will output the actual stage positions. For now, stage linearity is assumed. """
    
    zMin = shearStagePosition[0]
    zMax = shearStagePosition[-1]
    
    movingMirrorZ = np.linspace(zMin,zMax,len(shearStagePosition))
    
    for ii in range(movingMirrorData.shape[1]):
        
        movingMirrorData[:,ii] = np.interp(movingMirrorZ, shearStagePosition, movingMirrorData[:,ii])
    
    zMin = twoDSIStagePosition[0]
    zMax = twoDSIStagePosition[-1]
    
    tdsiTraceZ = np.linspace(zMin,zMax,len(twoDSIStagePosition))
    
    for ii in range(movingMirrorData.shape[1]):
        
        tdsiTrace[:,ii] = np.interp(tdsiTraceZ, twoDSIStagePosition, tdsiTrace[:,ii])
    
    
    
    
    #%% Extract vector data form matrices
    wavelengths = wavelengths * 1e-9
    upconvPowerSpectrum = fixedMirrorData
    
    
    if spectrum is not None:
        data2 = np.load(folder+spectrum)
        fundPowerSpectrum = data2['spectrum']
        fundWavelengths = data2['wavelengths']*1e-9
        data2.close()
        if upconvWavelength == 'auto':
            upconvWavelength,downConvWavelengths, downConvSpectrum = library_2dsi.find_upconversion(wavelengths, upconvPowerSpectrum, fundWavelengths, fundPowerSpectrum)
            print('Upconversion wavelength: ' + str(round(upconvWavelength*1e9)) + ' nm')
        plt.figure()
        plt.plot(fundWavelengths*1e9,fundPowerSpectrum/fundPowerSpectrum.max(), label = 'Fundamental spectrum')
        plt.plot(downConvWavelengths*1e9, downConvSpectrum/downConvSpectrum.max(),'--r', label = 'Un-upconverted spectrum')
        plt.ylim([0,1.05])
        plt.xlabel('Wavelengths [nm]')
        plt.ylabel('Normalized power density')
        plt.legend()
        
        
    if upconvWavelength == 'auto':
    
        upconvWavelength = wavelengths[np.argmax(upconvPowerSpectrum)]*2
        print('Upconversion wavelength: ' + str(round(upconvWavelength*1e9)) + ' nm')
        
  
    
    """ Find mean shear of data """ 
    shearFrequency = library_2dsi.find_shear(wavelengths, upconvPowerSpectrum, movingMirrorData,movingMirrorZ,tdsiTraceZ,debug)
    
    """ Perform 1D Fourier transform of trace along stage position dimension """ 
    FFTspatialFreq, FFTamplitude, FFTphase = library_2dsi.make1Dfft(wavelengths,tdsiTraceZ,tdsiTrace,zeropadding = zeroPadTrace, windowing = windowTrace, debugGraphs=debug)
    
    """ Cut FFT data to region with minimum amplitude """ 
    FFTamplitude, FFTphase, FFTwavelengths = library_2dsi.cut_fft(FFTamplitude, FFTphase, wavelengths,debug)
    
    if spectrum is not None:
        
        FFTwavelengths = 1 / ( 1/FFTwavelengths - 1/upconvWavelength )
        wavelengths = fundWavelengths
        upconvPowerSpectrum = fundPowerSpectrum
        upconvWavelength = np.infty
    
    """ Calculate spectral phase from FFT. Currently only using concatenation algorithm.
        Also outputs GDD and TOD (this calculation wil be moved to a new function 
        once other algorithms will be implemented """ 
    concW, concPhase, midpointW, midpointPhase = library_2dsi.calc_spectral_phase(shearFrequency, upconvWavelength,FFTamplitude, FFTphase, FFTwavelengths,debugGraphs = debug)
    
    """ Calculate temporal enveloppe of pulse from upconverted power spectrum and
        spectral phase calculated above. Function directly outputs the square of 
        the enveloppe, to obtain intensity-like vertical units """
    tConc, pulseConc,Econc = library_2dsi.calc_temporal_envelope(wavelengths, upconvPowerSpectrum, upconvWavelength, concW, concPhase, True, minTimeResolution)
    tMidpoint, pulseMidpoint, Emidpoint = library_2dsi.calc_temporal_envelope(wavelengths, upconvPowerSpectrum, upconvWavelength, midpointW, midpointPhase, True,minTimeResolution)
    
    """ Calculate temporal enveloppe of Fourier-limited pulse """
    tLim, pulseLim, Elim = library_2dsi.calc_temporal_envelope(wavelengths, upconvPowerSpectrum, upconvWavelength, concW, np.zeros_like(concW), False,minTimeResolution)
    
    tInterp = np.linspace(tLim[1],tLim[-2],1000000)
    pulseConcInterp = interp(tConc,pulseConc,'quadratic',bounds_error = False,fill_value = 0)(tInterp)
    pulseMidpointInterp = interp(tMidpoint,pulseMidpoint,'quadratic',bounds_error = False,fill_value = 0)(tInterp)
    pulseLimInterp = interp(tLim,pulseLim,'quadratic',bounds_error = False,fill_value = 0)(tInterp)
    
    """ Realign pulse's peak intensity to t = 0 fs """
    tpeakConc = tInterp[ np.argmax(pulseConcInterp)]
    
    tpeakMidpoint = tInterp[ np.argmax(pulseMidpointInterp)]
    
    tpeakLim = tInterp[ np.argmax(pulseLimInterp)]
    
    if simulatedData:
        T = timeVector[-1] - timeVector[0]
        vIn, sIn = fq.ezfft(timeVector, inputPulse)
        phaseIn = np.unwrap(np.angle(sIn* np.exp(1j*2*np.pi*vIn*(T/2) ))) 
        p = np.polyfit(vIn, phaseIn,1, w = np.abs(sIn)**2)
        phaseIn -= np.polyval(p, vIn)
    
        pulseInInterp = interp(timeVector,np.abs(inputPulse)**2,'quadratic',bounds_error = False,fill_value = 0)(tInterp)
        tpeakIn = tInterp[ np.argmax(pulseInInterp)]
        inputPulse = interp(timeVector-tpeakIn, inputPulse,'quadratic',bounds_error = False,fill_value = 0)(timeVector)
        
        
        
    """ Make figures and output values in console. Will be cleaned up in next update """
    fig = plt.figure()
    ax = fig.gca()
    ax.plot((tConc-tpeakConc)*1e15, pulseConc/np.max(pulseConc), '--b', linewidth =3 , label = 'Concat.')
    ax.plot((tMidpoint-tpeakMidpoint)*1e15, pulseMidpoint/np.max(pulseMidpoint), '--r', linewidth =3, label = 'Midpoint')
    ax.plot((tLim-tpeakLim)*1e15, pulseLim/np.max(pulseLim), 'k', linewidth =3, label = 'Fourier-lim.')
    ax.set_ylim([0,1.05])
    ax.set_xlabel('Time [fs]')
    ax.set_ylabel('Normalised power')
    if simulatedData:
        ax.plot(timeVector*1e15, np.abs(inputPulse)**2 / np.max( np.abs(inputPulse)**2), '-.g', label = 'Input')
    ax.legend()
    ax.set_xlim([-150, 150])
    
    
    
    vp, sp = fq.ezfft(tConc,Econc)
    T = tConc[-1]-tConc[0]
    phaseC = np.unwrap(np.angle(sp* np.exp(1j*2*np.pi*vp*(T/2) ))) 
    p = np.polyfit(vp, phaseC,1, w = np.abs(sp)**2)
    phaseC -= np.polyval(p, vp)

    vm, sm = fq.ezfft(tMidpoint,Emidpoint)
    T = tMidpoint[-1]-tMidpoint[0]
    phaseM = np.unwrap(np.angle(sm* np.exp(1j*2*np.pi*vm*(T/2) ))) 
    p = np.polyfit(vm, phaseM,1, w = np.abs(sm)**2)
    phaseM -= np.polyval(p, vm)
    
    
    plt.figure()
    axLeft = plt.gca()
    axLeft.plot(vp[vp>0]/1e12, np.abs(sp[vp>0])**2 / np.max(np.abs(sp[vp>0])**2) , '-k', linewidth = 3)
    axRight = plt.twinx()
    axRight.plot(vp[ np.abs(sp)**2 > (np.abs(sp)**2).max()/20 ]/1e12, phaseC[ np.abs(sp)**2 > (np.abs(sp)**2).max()/20 ], '--b', linewidth = 3, label = 'Concatenation')
    axRight.plot(vm[ np.abs(sm)**2 > (np.abs(sm)**2).max()/20 ]/1e12, phaseM[ np.abs(sm)**2 > (np.abs(sm)**2).max()/20 ], '--r', linewidth = 3, label = 'Midpoint')
    if simulatedData:
        axLeft.plot(vIn[vIn>0]/1e12, np.abs(sIn[vIn>0])**2 / np.max(np.abs(sIn[vIn>0])**2) , '-g', linewidth = 2)
        axRight.plot(vIn[ np.abs(sIn)**2 > (np.abs(sIn)**2).max()/20 ]/1e12, phaseIn[ np.abs(sIn)**2 > (np.abs(sIn)**2).max()/20 ], '--g', linewidth = 2, label = 'Input')
    axLeft.set_xlim(C/(1 / (1/wavelengths - 1/upconvWavelength))[-1]/1e12,C/(1 / (1/wavelengths - 1/upconvWavelength))[0]/1e12)
    axLeft.set_xlabel('Frequency [Thz]')
    axLeft.set_ylabel('Power spectral density [arb. u.]')
    axRight.set_ylabel('Spectral phase [rad]')
    axRight.legend()
    axLeft.set_ylim([0,1.05])
    
    
    print("Shear frequency: " + str(round(shearFrequency/1e12,1)) +" THz")
    
    
    
    if  np.isnan(fq.ezfindwidth(tConc, pulseConc)):
        print( "Conc. pulse duration: NaN"  )
    else:
        print( "Conc. pulse duration: " + str(round(fq.ezfindwidth(tConc, pulseConc)*1e15*100)/100) +" fs" )
    
    if  np.isnan(fq.ezfindwidth(tMidpoint, pulseMidpoint)):
        print( "Midpoint pulse duration: NaN"  )
    else:
        print( "Midpoint pulse duration: " + str(round(fq.ezfindwidth(tMidpoint, pulseMidpoint)*1e15*100)/100) +" fs" )
        
    print( "Fourier-lim. pulse duration: " + str(round(fq.ezfindwidth(tLim, pulseLim)*1e15*100)/100) +" fs")

    if simulatedData:
        print( "Input pulse duration: " + str(round(fq.ezfindwidth(timeVector, np.abs(inputPulse)**2)*1e15*100)/100) +" fs")
    
    return tConc, Econc, Emidpoint
    
    
def shgFROG(filename, initialGuess = 'gaussian', tau = None, method = 'copra', dt = None , maxIter = 100, symmetrizeGrid = False, wavelengthLimits = [0,np.inf], gridSize = None, smoothTrace = True, relativeNoiseTreshold = 0, marginalCorrection = None,inputDelays = None, inputWavelengths = None, inputTrace = None,makeFigures = True):
    
    """
    Uses pypret retrieval module developped by Nils C. Geib at the Institute of Applied Physics of the University of Jena, Germany.
    It is available at https://github.com/ncgeib/pypret, and is licenced under the MIT License (see LICENSE file in the module).
    
    
    """
    
    if (inputDelays is not None) & (inputWavelengths is not None) & (inputTrace is not None):
         delays = inputDelays
         wavelengths  = inputWavelengths
         trace = inputTrace
    else:
         delays, wavelengths, trace = library_frog.unpack_data(filename,wavelengthLimits,makeFigures)

    """ Recenter the trace to zero delay. Otherwise copra behaves weirdly"""
    marginal_t = simpson(trace,wavelengths,axis = 1)
    t_0 = delays[np.argmax(marginal_t)]
    delays -= t_0
    
    if smoothTrace:
        for ii, delay in enumerate(delays):
            trace[ii,:] = fq.ezsmooth(trace[ii,:])
    
    """ Removing negative values from trace. Seems to give slightly better results"""
    trace[trace<trace.max()*relativeNoiseTreshold] = 0
    
    """ PCGPA algorithm requires a symmetric grid """
    if method.lower() == 'pcgpa':
        symmetrizeGrid = True
    
    """ Adjust grid size if required """
    if symmetrizeGrid:
        if gridSize is None:
            gridSize = [len(wavelengths),len(wavelengths)]
        else:
            gridSize[1] = gridSize[0]
        
    
    
    if dt is None:
        dt = np.mean(np.diff(delays))
        
    if gridSize is not None:
        
        symTrace = np.zeros((gridSize[0],len(wavelengths)))
        
        delayLim = np.min([abs(delays[0]),abs(delays[-1])])
        
        # Symmetrize delay grid wrt frequency grid
        symDelays = np.linspace(-delayLim,delayLim,gridSize[0])
        
        for ii, wavelength in enumerate(wavelengths):
            interpTrace = interp(delays,trace[:,ii],'quadratic',bounds_error=False,fill_value=0)
            symTrace[:,ii] = interpTrace(symDelays)
            symTrace[:,ii] = 0.5*interpTrace(np.hstack((symDelays[symDelays<=0],symDelays[symDelays<0][-1::-1]))) + 0.5*interpTrace(np.hstack((symDelays[symDelays>=0][-1::-1],symDelays[symDelays>0])))
            
        trace = symTrace
        delays = symDelays
        dt = np.mean(np.diff(delays))
    
    """ Define time/frequency grids for input pulse """
    if gridSize is not None:
        ft = pypret.FourierTransform(gridSize[1], dt = dt)
    else:
        ft = pypret.FourierTransform(len(delays), dt = dt)
    
    """ Integrate over delay axis"""
    marginal_w = simpson(trace,delays,axis = 0)
    
        
    """ Carrier wavelength """
    lambda_0 = C/(simpson(C/wavelengths[-1::-1]*marginal_w[-1::-1],C/wavelengths[-1::-1],axis = 0)/simpson(marginal_w[-1::-1],C/wavelengths[-1::-1],axis = 0)) * 2
    
    """ Marginal correction: compare frequnecy marginal to spectrum autoconvolution.
        Relative differences between the two should correspond to experimental
        bandwidth limitation. Trace is adjusted accordingly to offset this effect. """
    if marginalCorrection is not None:
        data = np.load(marginalCorrection)
        corrWavelengths = data['wavelengths']*1e-9
        corrSpectrum = data['spectrum']
        data.close()
        
        corrSpectrumRaw = np.copy(corrSpectrum)
        
        corrW = np.linspace(-4*np.pi*C/corrWavelengths[0],4*np.pi*C/corrWavelengths[0],4*len(corrWavelengths)+1)
        
        corrSpectrum = interp( 2*np.pi*C/corrWavelengths[-1::-1], corrSpectrum[-1::-1]*corrWavelengths[-1::-1]**2/C ,bounds_error=False,fill_value=0)(corrW)
        
        x,y = fq.ezifft(corrW,corrSpectrum)
        absc_conv,autoConv = fq.ezfft(x,y**2,neg = True) 
        autoConv = np.real(autoConv)

        
        autoConv = interp(absc_conv, autoConv*absc_conv**2/C,bounds_error=False,fill_value=0)(2*np.pi*C/wavelengths)
        
        marginal_w_corr = np.copy(marginal_w)
        marginal_w_corr[marginal_w_corr<=0] = marginal_w_corr[marginal_w_corr>0].min()
        marginal_w_corr = fq.ezsmooth(marginal_w_corr,15,'hanning')
        marginalCorr = ( autoConv/autoConv.max() ) / ( marginal_w_corr / marginal_w_corr.max() )

        for ii, delay in enumerate(delays):
            trace[ii,:]*=marginalCorr
        if makeFigures:
            plt.figure()
            plt.plot(wavelengths*1e9,autoConv/autoConv.max(),label = 'From spectrum')
            plt.plot(wavelengths*1e9,marginal_w_corr / marginal_w_corr.max(), label = 'From FROG trace')
            plt.xlabel('Wavelengths [nm]')
            plt.ylabel('Frequency margianal')
            plt.ylim([0,1.05])
            plt.legend()
            
            plt.figure()
            plt.plot(wavelengths*1e9,marginalCorr)
            plt.xlabel('Wavelengths [nm]')
            plt.ylabel('Marginal correction factor')
            plt.ylim(bottom=0)
    
    """ Instantiate a pulse object w/ appropriate carrier wavelength
        (other parameters don't matter here)"""
    pulseP = pypret.Pulse(ft, lambda_0)
    pypret.random_gaussian(pulseP, 1e-15, phase_max=0.0)
        
    """ Instantiate a PNPS object for SHG-FROG technique"""
    pnps = pypret.PNPS(pulseP, "frog", "shg")
    pnps.calculate(pulseP.spectrum, delays)
    
    """ Export SHG frequency grid """
    w_shg = pnps.process_w
    w_fund = pulseP.w+pulseP.w0
    
    
    
    
    """ Interpolate trace to shg frequency grid """
    trace_w = np.zeros((len(delays),len(w_shg)))
    for ii,delay in enumerate(delays):  
    
        interpTrace = interp(C/wavelengths[-1::-1],trace[ii,:][-1::-1],'quadratic',bounds_error=False,fill_value=0)
        trace_w[ii,:] = interpTrace(w_shg/2/np.pi) * C/(w_shg/2/np.pi)**2
        
        
    
    
    """ Plot interpolated trace (to check interpolation errors) """
    if makeFigures:
        plt.figure()
        plt.pcolormesh(delays*1e15,(2*np.pi*C/w_shg[w_shg>0])*1e9,trace_w[:,w_shg>0].transpose() / (trace_w[:,w_shg>0].transpose()).max() ,shading='auto')
        if marginalCorrection is None:
            plt.title('Input trace (interpolated)')
        else:
            plt.title('Input trace (corrected + interpolated)')
        plt.xlabel('Delay [fs]')
        plt.ylabel('Wavelengths [nm]')
        plt.ylim(wavelengths[0]*1e9,wavelengths[-1]*1e9)
        plt.colorbar()

    """ Reformat trace for retriever """
    traceInput = pypret.mesh_data.MeshData(trace_w,delays,w_shg,labels = ['Delay','Frequency',''])


    """ Initial guess for iterative algorithm. Three options:
        Gaussian (default): Fits a gaussian pulse to both the time & freq. marginals. Struggles w/ non-bell-shaped spectra.
        Spectrum: Takes the independantly measured spectrum as initial guess with flat phase.
        RANA: Uses "RANA" algorithm to deduce the spectrum from the trace. Uses flat phase. Struggles w/ noise."""
        
    if initialGuess.lower() == 'gaussian':
        if tau is None:
            autocorr =  simpson(trace,wavelengths,axis = 1)
            tau = library_frog.get_FWHM(delays,autocorr)/np.sqrt(2) /np.sqrt(2*np.log(2))
            
            dw = library_frog.get_FWHM(2*np.pi*C/wavelengths[-1::-1],marginal_w[-1::-1])/np.sqrt(2) /np.sqrt(2*np.log(2))
            tau_0 = 2 / (dw)
            
            if tau > tau_0:
                GDD = (tau**2*tau_0**2 - tau_0**4)**0.5/2
            else:
                GDD = 0
        else:
            dw = 2  / (tau/np.sqrt(2*np.log(2))) 
            
        
        w_0 = 2*np.pi*C/lambda_0
        initialGuess = np.complex128(np.exp(- (w_fund-w_0)**2 / dw**2)) * np.exp(1j*GDD*(w_fund-w_0)**2)
    
    elif (initialGuess.lower()=='spectrum') & (marginalCorrection is not None):
       initialGuess = interp( 2*np.pi*C/corrWavelengths[-1::-1], corrSpectrumRaw[-1::-1]*corrWavelengths[-1::-1]**2/C  ,bounds_error=False, fill_value=0)(w_fund)
       initialGuess[initialGuess<0]=0
       initialGuess = np.complex128(initialGuess**0.5)
       initialGuess /= initialGuess.max()
    else:
        initialGuess = library_frog.RANA(delays,w_shg,trace_w,w_fund)
    
    
    """ Instantiate retriever """
    ret = pypret.Retriever(pnps,method =  method, verbose=True, maxiter=maxIter)


    """ Apply retrieval algorithm and print results """
    ret.retrieve(traceInput, initialGuess)
    results = ret.result()
    
    """ Export retrieved pulse & trace """
    pulseRetrieved = results.pulse_retrieved
    traceRetrieved = results.trace_retrieved
    pulseFrequencies = w_fund/(2*np.pi)
    traceFrequencies = w_shg/(2*np.pi)
    
    """ Make plots """
    if makeFigures:
        axSpectrum = library_frog.plot_output(pulseRetrieved, initialGuess, pulseFrequencies, traceRetrieved, traceFrequencies,delays, wavelengths)
    
        if marginalCorrection is not None:
            refSpectrum = corrSpectrumRaw * corrWavelengths**2/C
            refSpectrum /= refSpectrum.max()
            refSpectrum = np.interp(pulseFrequencies[pulseFrequencies>0], C/corrWavelengths[-1::-1],refSpectrum[-1::-1])
            axSpectrum.plot(pulseFrequencies[pulseFrequencies>0] / 1e12,refSpectrum,'g--',linewidth = 3,label = 'Measured')
            axSpectrum.set_ylim([0,1.05])
        axSpectrum.legend()
    
    return pulseRetrieved, initialGuess, pulseFrequencies, traceRetrieved, traceFrequencies,delays, wavelengths


def shgDscan(filename, initialGuess = 'gaussian', tau = None, method = 'copra', dt = None , maxIter = 100, symmetrizeGrid = False, wavelengthLimits = [0,np.inf], gridSize = None, marginalCorrection = None, inputDelays = None, inputWavelengths = None, inputTrace = None, makeFigures = True):
    
    
    delays, wavelengths, trace = library_dscan.unpack_data(filename,wavelengthLimits)           # Here delay is actually insertion
                                                                                                # It was kept for simplicity's sake
    # """ Recenter the trace to zero delay. Otherwise copra behaves weirdly"""
    # marginal_t = simpson(trace,wavelengths,axis = 1)                                    # Removed because a dscan trace centered
    # t_0 = delays[np.argmax(marginal_t)]                                                 # at 0 loses a lot of information
    # delays -= t_0
    
    """ Removing negative values from trace. Seems to give slightly better results"""
    trace[trace<0] = 0
    
    """ PCGPA algorithm requires a symmetric grid """
    if method.lower() == 'pcgpa':
        symmetrizeGrid = True
    
    """ Adjust grid size if required """
    if symmetrizeGrid:
        if gridSize is None:
            gridSize = [len(wavelengths),len(wavelengths)]
        else:
            gridSize[1] = gridSize[0]
        
    
    
    if dt is None:
        dt = np.mean(np.diff(delays))
        
    if gridSize is not None:
        
        symTrace = np.zeros((gridSize[0],len(wavelengths)))
        
        delayLim = np.min([abs(delays[0]),abs(delays[-1])])
        
        # Symmetrize delay grid wrt frequency grid
        symDelays = np.linspace(-delayLim,delayLim,gridSize[0])
        
        for ii, wavelength in enumerate(wavelengths):
            interpTrace = interp(delays,trace[:,ii],'quadratic',bounds_error=False,fill_value=0)
            symTrace[:,ii] = interpTrace(symDelays)
            symTrace[:,ii] = 0.5*interpTrace(np.hstack((symDelays[symDelays<=0],symDelays[symDelays<0][-1::-1]))) + 0.5*interpTrace(np.hstack((symDelays[symDelays>=0][-1::-1],symDelays[symDelays>0])))
            
        trace = symTrace
        delays = symDelays
        dt = np.mean(np.diff(delays))
    
    """ Define time/frequency grids for input pulse """
    if gridSize is not None:
        ft = pypret.FourierTransform(gridSize[1], dt = dt)
    else:
        ft = pypret.FourierTransform(len(delays), dt = dt)
    
    """ Integrate over delay axis"""
    marginal_w = simpson(trace,delays,axis = 0)
    
        
    """ Carrier wavelength """
    lambda_0 = C/(simpson(C/wavelengths[-1::-1]*marginal_w[-1::-1],C/wavelengths[-1::-1],axis = 0)/simpson(marginal_w[-1::-1],C/wavelengths[-1::-1],axis = 0)) * 2
    
    """ Marginal correction: compare frequnecy marginal to spectrum autoconvolution.
        Relative differences between the two should correspond to experimental
        bandwidth limitation. Trace is adjusted accordingly to offset this effect. """
    if marginalCorrection is not None:
        data = np.load(marginalCorrection)
        corrWavelengths = data['wavelengths']*1e-9
        corrSpectrum = data['spectrum']
        
        corrSpectrumRaw = np.copy(corrSpectrum)
        
        corrW = np.linspace(-4*np.pi*C/corrWavelengths[0],4*np.pi*C/corrWavelengths[0],4*len(corrWavelengths)+1)
        
        corrSpectrum = interp( 2*np.pi*C/corrWavelengths[-1::-1], corrSpectrum[-1::-1] ,bounds_error=False,fill_value=0)(corrW)
        
        x,y = fq.ezifft(corrW,corrSpectrum)
        absc_conv,autoConv = fq.ezfft(x,y**2,neg = True) 
        autoConv = np.real(autoConv)

        
        autoConv = interp(absc_conv, autoConv,bounds_error=False,fill_value=0)(2*np.pi*C/wavelengths)
        
        marginal_w_corr = np.copy(marginal_w)
        marginal_w_corr[marginal_w_corr<=0] = marginal_w_corr[marginal_w_corr>0].min()
        marginal_w_corr = fq.ezsmooth(marginal_w_corr,15,'hanning')
        marginalCorr = ( autoConv/autoConv.max() ) / ( marginal_w_corr / marginal_w_corr.max() )

        for ii, delay in enumerate(delays):
            trace[ii,:]*=marginalCorr
        
        plt.figure()
        plt.plot(wavelengths*1e9,autoConv/autoConv.max(),label = 'From spectrum')
        plt.plot(wavelengths*1e9,marginal_w_corr / marginal_w_corr.max(), label = 'From FROG trace')
        plt.xlabel('Wavelengths [nm]')
        plt.ylabel('Frequency margianal')
        plt.ylim([0,1.05])
        plt.legend()
        
        plt.figure()
        plt.plot(wavelengths*1e9,marginalCorr)
        plt.xlabel('Wavelengths [nm]')
        plt.ylabel('Marginal correction factor')
        plt.ylim(bottom=0)
    
    """ Instantiate a pulse object w/ appropriate carrier wavelength
        (other parameters don't matter here)"""
    pulseP = pypret.Pulse(ft, lambda_0)
    pypret.random_gaussian(pulseP, 1e-15, phase_max=0.0)
        
    """ Instantiate a PNPS object for SHG-FROG technique"""
    pnps = pypret.PNPS(pulseP, "dscan", "shg")
    pnps.calculate(pulseP.spectrum, delays)
    
    """ Export SHG frequency grid """
    w_shg = pnps.process_w
    w_fund = pulseP.w+pulseP.w0

    
    
    
    """ Interpolate trace to shg frequency grid """
    trace_w = np.zeros((len(delays),len(w_shg)))
    for ii,delay in enumerate(delays):  
    
        interpTrace = interp(C/wavelengths[-1::-1],trace[ii,:][-1::-1],'quadratic',bounds_error=False,fill_value=0)
        trace_w[ii,:] = interpTrace(w_shg/2/np.pi)
        
        
    
    
    """ Plot interpolated trace (to check interpolation errors) """
    plt.figure()
    plt.pcolormesh(delays*1e15,(2*np.pi*C/w_shg)*1e9,trace_w.transpose())
    if marginalCorrection is None:
        plt.title('Input trace (interpolated)')
    else:
        plt.title('Input trace (corrected + interpolated)')
    plt.xlabel('Delay [fs]')
    plt.ylabel('Wavelengths [nm]')
    plt.ylim(wavelengths[0]*1e9,wavelengths[-1]*1e9)
    plt.colorbar()

    """ Reformat trace for retriever """
    traceInput = pypret.mesh_data.MeshData(trace_w,delays,w_shg,labels = ['Delay','Frequency',''])


    """ Initial guess for iterative algorithm. Three options:
        Gaussian (default): Fits a gaussian pulse to both the time & freq. marginals. Struggles w/ non-bell-shaped spectra.
        Spectrum: Takes the independantly measured spectrum as initial guess with flat phase.
        RANA: Uses "RANA" algorithm to deduce the spectrum from the trace. Uses flat phase. Struggles w/ noise."""
        
    if initialGuess.lower() == 'gaussian':
        if tau is None:
            autocorr =  simpson(trace,wavelengths,axis = 1)
            tau = library_dscan.get_FWHM(delays,autocorr)/np.sqrt(2) /np.sqrt(2*np.log(2))
            
            dw = library_dscan.get_FWHM(2*np.pi*C/wavelengths[-1::-1],marginal_w[-1::-1])/np.sqrt(2) /np.sqrt(2*np.log(2))
            tau_0 = 2 / (dw)
            
            if tau > tau_0:
                GDD = (tau**2*tau_0**2 - tau_0**4)**0.5/2
            else:
                GDD = 0
        else:
            dw = 2  / (tau/np.sqrt(2*np.log(2))) 
            
        
        w_0 = 2*np.pi*C/lambda_0
        initialGuess = np.complex128(np.exp(- (w_fund-w_0)**2 / dw**2)) * np.exp(1j*GDD*(w_fund-w_0)**2)
    
    elif (initialGuess.lower()=='spectrum') & (marginalCorrection is not None):
       initialGuess = interp( 2*np.pi*C/corrWavelengths[-1::-1], corrSpectrumRaw[-1::-1] ,bounds_error=False, fill_value=0)(w_fund)
       initialGuess[initialGuess<0]=0
       initialGuess = np.complex128(initialGuess**0.5)
       initialGuess /= initialGuess.max()
    else:
        initialGuess = library_dscan.RANA(delays,w_shg,trace_w,w_fund)
    
    
    """ Instantiate retriever """
    ret = pypret.Retriever(pnps,method =  method, verbose=True, maxiter=maxIter)


    """ Apply retrieval algorithm and print results """
    ret.retrieve(traceInput, initialGuess)
    results = ret.result()
    
    """ Export retrieved pulse & trace """
    pulseRetrieved = results.pulse_retrieved
    traceRetrieved = results.trace_retrieved
    pulseFrequencies = w_fund/(2*np.pi)
    traceFrequencies = w_shg/(2*np.pi)
    
    """ Make plots """
    axSpectrum = library_dscan.plot_output(pulseRetrieved, initialGuess, pulseFrequencies, traceRetrieved, traceFrequencies,delays, wavelengths)
    
    if marginalCorrection is not None:
        axSpectrum.plot(corrWavelengths*1e9,corrSpectrumRaw/corrSpectrumRaw.max(),'g--',linewidth = 3,label = 'Measured')
        axSpectrum.set_ylim([0,1.05])
    axSpectrum.legend()
    
    return pulseRetrieved, initialGuess, pulseFrequencies, traceRetrieved, traceFrequencies,delays, wavelengths
