

#%% Import modules
import femtoQ.tools as fq
import numpy as np
import matplotlib.pyplot as plt
from femtoQ.pr_backend import tdsilib as library_2dsi
from femtoQ.pr_backend import froglib as library_frog
from scipy.constants import c as C
from scipy.interpolate import interp1d as interp
from scipy.integrate import simps as simpson
import pypret



def twodsi(filename,  upconvWavelength = 'auto', wavelengthCutoffs = None, smoothSpectrum = False, zeroPadTrace = True, windowTrace = True, minTimeResolution = 1e-15, folder = ''):

    #%% Analysis parameters
    debug = False                # Setting True will output many more figures for debugging
    simulatedData = False        # Setting to True will load last set of simulated data rather than experimental data
    fitPolynomialToPhase = False # Setting to an integer value >1 will fit a polynomial of same order to the phase curve in w space
    simulateFrogTrace = False    # Simulate a Frog trace with final result
    
    
    
    
    """ If file contains data acquired over a long distance on the stage, with a shear
        that changes a lot from start to end, set a value here to window the trace
        around the point where the shear is equal to that value """
    manualShear = None
    
    
    """ Range of stage displacement over which to window the trace. 
        Set to None to keep full data. """
    zWindowLength = None
    
    data = np.load(folder+filename)
    fixedMirrorData = data['upconvSpectrum']
    wavelengths = data['wavelengths']
    shearStagePosition = data['shearStagePosition']
    movingMirrorData = data['shearTrace']
    twoDSIStagePosition = data['twoDSIStagePosition']
    tdsiTrace = data['twoDSITrace']
    
    #%% Load data
    if simulatedData:
        tdsiTrace = data['twoDSITrace']
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
    fixedMirrorData, movingMirrorData = library_2dsi.substractNoiseFloor(fixedMirrorData, movingMirrorData)
    
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
    
    if upconvWavelength == 'auto':
    
        upconvWavelength = wavelengths[np.argmax(upconvPowerSpectrum)]*2
        
        
    #%% Smooth spectrum, if desired
    # =============================================================================
    # if smoothingSpectrum:
    #     upconvPowerSpectrum = fq.ezsmooth(upconvPowerSpectrum, window_len = 7, window = 'flat')
    #     for ii in range(movingMirrorData.shape[0]):
    #         movingMirrorData[ii,:] = fq.ezsmooth(movingMirrorData[ii,:], window_len = 7, window = 'flat')
    # 
    # =============================================================================
    #%% Define linear space of stage displacement
    
    #%% Set trace windowing parameters
    if manualShear is not None:
    
        # Make a map of shear vs stage position
        shearMap = library_2dsi.make_shear_map(wavelengths, upconvPowerSpectrum, movingMirrorData,movingMirrorZ,debug)
        
        # Make a linear fit
        p = np.polyfit(shearMap, movingMirrorZ,1)
        
        # Find position of desired shear
        z0 = np.polyval(p, manualShear)
        
        # Set center of window  to center of data if desired shear can't be obtained
        if (z0 > zMax) or (z0 < zMin):
            z0 = (zMax + zMin)/2
    else:
        z0 = (zMin+zMax)/2
    
    
    #%% Window the trace, if desired
    if zWindowLength is not None:
        zMin = z0 - zWindowLength/2
        zMax = z0 + zWindowLength/2
        tdsiTrace = tdsiTrace[((tdsiTraceZ>=zMin) & (tdsiTraceZ<=zMax)),:]
        tdsiTraceZ = tdsiTraceZ[((tdsiTraceZ>=zMin) & (tdsiTraceZ<=zMax))]
        movingMirrorData = movingMirrorData[((movingMirrorZ>=zMin) & (movingMirrorZ<=zMax)),:]
        movingMirrorZ = movingMirrorZ[((movingMirrorZ>=zMin) & (movingMirrorZ<=zMax))]
    
    
    """ Find mean shear of data """ 
    shearFrequency = library_2dsi.find_shear(wavelengths, upconvPowerSpectrum, movingMirrorData,movingMirrorZ,tdsiTraceZ,debug)
    
    """ Perform 1D Fourier transform of trace along stage position dimension """ 
    FFTspatialFreq, FFTamplitude, FFTphase = library_2dsi.make1Dfft(wavelengths,tdsiTraceZ,tdsiTrace,zeropadding = zeroPadTrace, windowing = windowTrace, debugGraphs=debug)
    
    """ Cut FFT data to region with minimum amplitude """ 
    FFTamplitude, FFTphase, FFTwavelengths = library_2dsi.cut_fft(FFTamplitude, FFTphase, wavelengths,debug)
    
    """ Calculate spectral phase from FFT. Currently only using concatenation algorithm.
        Also outputs GDD and TOD (this calculation wil be moved to a new function 
        once other algorithms will be implemented """ 
    concW, concPhase,concGD, concGDD, concTOD, midpointW, midpointPhase,midpointGD, midpointGDD, midpointTOD = library_2dsi.calc_spectral_phase(shearFrequency, upconvWavelength,FFTamplitude, FFTphase, FFTwavelengths,polyfit = fitPolynomialToPhase,debugGraphs = debug)
    
    """ Calculate temporal enveloppe of pulse from upconverted power spectrum and
        spectral phase calculated above. Function directly outputs the square of 
        the enveloppe, to obtain intensity-like vertical units """
    tConc, pulseConc,Econc = library_2dsi.calc_temporal_envelope(wavelengths, upconvPowerSpectrum, upconvWavelength, concW, concPhase, True, minTimeResolution)
    tMidpoint, pulseMidpoint, Emidpoint = library_2dsi.calc_temporal_envelope(wavelengths, upconvPowerSpectrum, upconvWavelength, midpointW, midpointPhase, True,minTimeResolution)
    
    """ Calculate temporal enveloppe of Fourier-limited pulse """
    tLim, pulseLim, Elim = library_2dsi.calc_temporal_envelope(wavelengths, upconvPowerSpectrum, upconvWavelength, concW, np.zeros_like(concW), False,minTimeResolution)
    
    """ Realign pulse's peak intensity to t = 0 fs """
    tpeak = tConc[ np.argmax(pulseConc)]
    pulseConc = np.interp(tConc, tConc-tpeak, pulseConc)
    
    tpeak = tMidpoint[ np.argmax(pulseMidpoint)]
    pulseMidpoint = np.interp(tMidpoint, tMidpoint-tpeak, pulseMidpoint)
    
    tpeak = tLim[ np.argmax(pulseLim)]
    pulseLim = np.interp(tLim, tLim-tpeak, pulseLim)
    
    if simulatedData:
        tpeak = timeVector[ np.argmax(inputPulse)]
        inputPulse = np.interp(timeVector, timeVector-tpeak, inputPulse)
    
    """ Make figures and output values in console. Will be cleaned up in next update """
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(tConc*1e15, pulseConc/np.max(pulseConc), '--b', label = 'Concat.')
    ax.plot(tMidpoint*1e15, pulseMidpoint/np.max(pulseMidpoint), '--r', label = 'Midpoint')
    ax.plot(tLim*1e15, pulseLim/np.max(pulseLim), 'k', label = 'Fourier lim.')
    ax.set_xlabel('Time [fs]')
    ax.set_ylabel('Normalised intensity')
    if simulatedData:
        ax.plot(timeVector*1e15, inputPulse, '-.g', label = 'Sim. input')
    ax.legend()
    ax.set_xlim([-150, 150])
    
# =============================================================================
#     fig = plt.figure()
#     ax = fig.gca()
#     ax.plot(tLim*1e15, pulseLim/np.max(pulseLim), 'k', label = 'Fourier limited')
#     ax.plot(tConc*1e15, pulseConc/np.max(pulseConc), '--r', label = 'Reconstructed')
#     ax.set_xlabel('Time [fs]')
#     ax.set_ylabel('Normalised intensity')
#     ax.legend()
#     ax.set_xlim([-150, 150])
# =============================================================================
    
    
    fig = plt.figure()
    axLeft = fig.gca()
    IIphase = np.argsort(concW,axis=None)
    IIGDD = np.argsort(concW[:,1:-1],axis=None)
    IITOD = np.argsort(concW[:,2:-2],axis=None)
    lambdaPhase = np.flip( 2*np.pi*2.998e8/concW.flatten()[IIphase]*1e9 )
    lambdaGDD = np.flip( 2*np.pi*2.998e8/concW[:,1:-1].flatten()[IIGDD]*1e9 )
    lambdaTOD = np.flip( 2*np.pi*2.998e8/concW[:,2:-2].flatten()[IITOD]*1e9 )
    concPhase =  np.flip( concPhase.flatten()[IIphase] )
    concGD =  np.flip( concGD.flatten()[IIGDD]*1e15 )
    concGDD =  np.flip( concGDD.flatten()[IIGDD]*1e30 )
    concTOD =  np.flip( concTOD.flatten()[IITOD]*1e45 )
    
    IIplot = upconvPowerSpectrum > np.max(upconvPowerSpectrum)/100
    wav1 = (1 / (1/wavelengths - 1/upconvWavelength)*1e9)[IIplot][0]
    wav2 = (1 / (1/wavelengths - 1/upconvWavelength)*1e9)[IIplot][-1]
    IIGDD = ((lambdaGDD>=wav1) & (lambdaGDD<=wav2))
    IIphase = ((lambdaPhase>=wav1) & (lambdaPhase<=wav2))
    w1 = 2*np.pi*2.998e8/wav2*1e9
    w2 = 2*np.pi*2.998e8/wav1*1e9
    
    
    axLeft.plot(1 / (1/wavelengths - 1/upconvWavelength)*1e9, upconvPowerSpectrum/np.max(upconvPowerSpectrum), '-b', linewidth = 2)
    axRight = axLeft.twinx()
    axRight.plot(lambdaGDD[IIGDD], concGD[IIGDD], '--r', linewidth = 2, label = 'Concat.')
    axRight.plot(2*np.pi*2.998e8/midpointW[1:-1][((midpointW[1:-1]>w1) & (midpointW[1:-1]<w2))]*1e9, midpointGD[((midpointW[1:-1]>w1) & (midpointW[1:-1]<w2))]*1e15, '--k', linewidth = 2, label = 'Midpoint')
    axLeft.set_xlabel('Wavelength [nm]')
    axLeft.set_ylabel('Normalized intensity')
    axRight.set_ylabel(r'Group delay [fs]')
    axRight.legend()
    
    fig2 = plt.figure()
    axLeft2 = fig2.gca()
    axLeft2.plot(1 / (1/wavelengths - 1/upconvWavelength)*1e9, upconvPowerSpectrum / np.max(upconvPowerSpectrum), '-b', linewidth = 2)
    axRight2 = axLeft2.twinx()
    axRight2.plot(lambdaPhase[IIphase], concPhase[IIphase], '--r', linewidth = 2)
    axLeft2.set_xlabel('Wavelength [nm]')
    axLeft2.set_ylabel('Normalized power spectrum')
    axRight2.set_ylabel(r'Spectral phase [rad]')
    axLeft2.yaxis.label.set_color('blue')
    axRight2.yaxis.label.set_color('red')
    axRight2.set_ylim(-np.pi,np.pi)
    
    weightsGDD = np.interp( lambdaGDD, 1 / (1/wavelengths - 1/upconvWavelength)*1e9, upconvPowerSpectrum )
    averageGDD = np.average(concGDD, weights =  weightsGDD)
    weightsTOD = np.interp( lambdaTOD, 1 / (1/wavelengths - 1/upconvWavelength)*1e9, upconvPowerSpectrum )
    averageTOD = np.average(concTOD, weights =  weightsTOD)
    
    
    
    
    
    if  np.isnan(fq.ezfindwidth(tConc, pulseConc)):
        print( "Conc. pulse duration: NaN"  )
    else:
        print( "Conc. pulse duration: " + str(round(fq.ezfindwidth(tConc, pulseConc)*1e15*100)/100) +" fs" )
    
    if  np.isnan(fq.ezfindwidth(tMidpoint, pulseMidpoint)):
        print( "Midpoint pulse duration: NaN"  )
    else:
        print( "Midpoint pulse duration: " + str(round(fq.ezfindwidth(tMidpoint, pulseMidpoint)*1e15*100)/100) +" fs" )
        
    print( "F. lim. pulse duration: " + str(round(fq.ezfindwidth(tLim, pulseLim)*1e15*100)/100) +" fs")
    print('Average GDD (conc.): ' + str(round(averageGDD*100)/100) + ' fs sq.') 
    print('Average TOD (conc.): ' + str(round(averageTOD*100)/100) + ' fs cu.') 
    
    print('Shear frequency:' + str(round(shearFrequency/1e12 * 100)/100) + ' THz')
    
    if simulateFrogTrace:
        library_2dsi.simFrogTrace(tConc,Econc,minWavelength,maxWavelength)
    
    plt.show()
    
    return
    
    
def shgFROG(filename, initialGuess = 'gaussian', tau = None, method = 'copra', dt = None , maxIter = 100, symmetrizeGrid = False, wavelengthLimits = [0,np.inf], gridSize = None):
    
    
    delays, wavelengths, trace = library_frog.unpack_data(filename,wavelengthLimits)
    
    marginal_t = simpson(trace,wavelengths,axis = 1)
    t_0 = t[np.argmax(marginal_t)]
    delays -= t_0
    
    
    if method.lower() == 'pcgpa':
        symmetrizeGrid = True
    
    
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
    
    # Define time/frequency grids for input pulse
    if gridSize is not None:
        ft = pypret.FourierTransform(gridSize[1], dt = dt)
    else:
        ft = pypret.FourierTransform(len(wavelengths), dt = dt)
    
    # Integrate over delay axis
    marginal_w = simpson(trace,delays,axis = 0)
    
    lambda_0 = C/(simpson(C/wavelengths[-1::-1]*marginal_w[-1::-1],C/wavelengths[-1::-1],axis = 0)/simpson(marginal_w[-1::-1],C/wavelengths[-1::-1],axis = 0)) * 2
    
    
    
    # Instantiate a pulse object w/ appropriate central wavelength
    # (other parameters don't matter here)
    pulseP = pypret.Pulse(ft, lambda_0)
    pypret.random_gaussian(pulseP, 1e-15, phase_max=0.0)
        
    # Instantiate a PNPS object for SHG-FROG technique
    pnps = pypret.PNPS(pulseP, "frog", "shg")
    pnps.calculate(pulseP.spectrum, delays)
    
    # Export SHG frequency grid
    w_shg = pnps.process_w
    w_fund = pulseP.w+pulseP.w0
    
    
    
    
    # Interpolate trace to shg frequency grid
    trace_w = np.zeros((len(delays),len(w_shg)))
    for ii,delay in enumerate(delays):  
    
        interpTrace = interp(C/wavelengths[-1::-1],trace[ii,:][-1::-1],'quadratic',bounds_error=False,fill_value=0)
        trace_w[ii,:] = interpTrace(w_shg/2/np.pi)
        
        
    
    
    # Plot interpolated trace (to check interpolation errors)
    plt.figure()
    plt.pcolor((2*np.pi*C/w_shg)*1e9,delays*1e15,trace_w)
    plt.title('Input trace (interpolated)')
    plt.ylabel('Delay [fs]')
    plt.xlabel('Wavelengths [nm]')
    plt.xlim(wavelengths[0]*1e9,wavelengths[-1]*1e9)
    plt.colorbar()

    # Reformat trace for retriever
    traceInput = pypret.mesh_data.MeshData(trace_w,delays,w_shg,labels = ['Delay','Frequency',''])


    # Apply RANA approach to get 1st estimate of input spectrum
    
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
        
    else:
        initialGuess = library_frog.RANA(delays,w_shg,trace_w,w_fund)
    
    
    # Instantiate retriver
    ret = pypret.Retriever(pnps,method =  method, verbose=True, maxiter=maxIter)


    # Apply retrieval algorithm and print results
    ret.retrieve(traceInput, initialGuess)
    results = ret.result()
    
    # Export retrieved pulse & trace
    pulseRetrieved = fq.ezsmooth(results.pulse_retrieved, window = 'hanning')
    traceRetrieved = results.trace_retrieved
    pulseFrequencies = w_fund/(2*np.pi)
    traceFrequencies = w_shg/(2*np.pi)
    
    library_frog.plot_output(pulseRetrieved, initialGuess, pulseFrequencies, traceRetrieved, traceFrequencies,delays, wavelengths)

    return pulseRetrieved, initialGuess, pulseFrequencies, traceRetrieved, traceFrequencies,delays, wavelengths
