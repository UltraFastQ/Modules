# -*- coding: utf-8 -*-
"""
Created on Sat May 25 20:50:09 2019

@author: Patrick
"""

import zipfile
import numpy as np
import femtoQ.tools as fq
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.integrate as integrate
import scipy.interpolate as interp


C = 299792458

def NextPowerOfTwo(number):
    """ Returns next power of two following 'number' """
    return int(np.ceil(np.log2(number)))

def unzip(folder,filename, tempUnzipDir):
    """ Unzip content of Labview's output zipped data file  """
    
    with zipfile.ZipFile(folder+filename,"r") as zip_ref:
        zip_ref.extractall(folder+tempUnzipDir)
    
    return 0



def import_data(folder,filename, tempUnzipDir):
    """ Load data from binary files extracted from Labview
    zip file """
    
    binFilenames = [ folder+tempUnzipDir+'/'+filename[0:-4]+'_fixed.bin', folder+tempUnzipDir+'/'+filename[0:-4]+'_shear.bin', folder+tempUnzipDir+'/'+filename[0:-4]+'_scan.bin']
        
    N = np.fromfile(binFilenames[0], count = 2, dtype='>i')
    fixedMirrorData = np.fromfile(binFilenames[0], dtype='>d')[1:]
    fixedMirrorData = np.reshape(fixedMirrorData, newshape = ((N[0],N[1])))
    
    
    movingMirrorData = np.fromfile(binFilenames[1], dtype='>i')
    movingMirrorData = np.reshape(movingMirrorData[2:], newshape = (movingMirrorData[0],movingMirrorData[1]))
    movingMirrorData = movingMirrorData[:,1:]
    
    tdsiTrace = np.fromfile(binFilenames[2], dtype='>i')
    tdsiTrace = np.reshape(tdsiTrace[2:], newshape = (tdsiTrace[0],tdsiTrace[1]))
    tdsiTrace = tdsiTrace[:,1:]

    return fixedMirrorData, movingMirrorData, tdsiTrace


def cut_spectrum(minWavelength, maxWavelength, fixedMirrorData, movingMirrorData, wavelengths, tdsiTrace):
    """ Cut spectra to relevant wavelengths region """
    
    ii1 = np.argmin( np.abs(wavelengths - (minWavelength * 1e9 )))
    ii2 = np.argmin( np.abs(wavelengths - (maxWavelength * 1e9 )))


    fixedMirrorData = fixedMirrorData[ii1:ii2+1]
    movingMirrorData = movingMirrorData[:,ii1:ii2+1]
    wavelengths = wavelengths[ii1:ii2+1]
    tdsiTrace = tdsiTrace[:,ii1:ii2+1]
    
    return fixedMirrorData, movingMirrorData, wavelengths, tdsiTrace


def substractNoiseFloor(fixedMirrorData, movingMirrorData):
    """ Substract noise floor from spectra (except trace) """
    
    lowLimit = 0.01
    
    fixedMirrorData = fixedMirrorData - np.max((fixedMirrorData[0],fixedMirrorData[-1]))
    fixedMirrorData[fixedMirrorData<0] = 0
    maxFixed = np.max(fixedMirrorData)
    fixedMirrorData[ fixedMirrorData < maxFixed*lowLimit] = 0
    
    for ii in range(movingMirrorData.shape[0]):
        movingMirrorData[ii,:] = movingMirrorData[ii,:] - np.max((movingMirrorData[ii,0],movingMirrorData[ii,-1]))
        movingMirrorData[ii,:][movingMirrorData[ii,:]<0] = 0
        maxMoving = np.max(movingMirrorData[ii,:])
        movingMirrorData[ii,:][ movingMirrorData[ii,:] < maxMoving*lowLimit] = 0
    
    
    return fixedMirrorData, movingMirrorData


def find_shear(wavelengths, upconvPowerSpectrum, movingMirrorData, movingMirror_Z,tdsiTraceZ,debugGraphs):
    """ Calculate shear frequency as a function of stage position,
    and take its value at the middle position as constant approximation """
    
    # Normalize spectra to max of one
    upconvPowerSpectrum = upconvPowerSpectrum / np.max(upconvPowerSpectrum)
    
    maxValues = np.max(movingMirrorData, axis = 1)
    tmp = wavelengths.shape[0]
    maxValues = np.transpose( np.tile(maxValues, (tmp,1)) )
    movingMirrorData = movingMirrorData / maxValues
    


    # Convert wavelengths to frequencies
    frequencies = C / wavelengths
    
    frequencies = np.flip(frequencies) # Flipping from low to high frequencies
    upconvPowerSpectrum = np.flip(upconvPowerSpectrum)
    movingMirrorData = np.flip(movingMirrorData,axis = 1)
    
    # Interpolate to a linear spacing of frequencies
    # Choice of datapoint position strongly affect results, here I am copying Matlab
    # need to check if another strategy would work better
    Df = frequencies[-1] - frequencies[0]
    df = np.max( np.diff(frequencies) ) / 16
    N = int(round(Df / df))
    linFreqs = np.linspace(frequencies[-1]-(N-1)*df, frequencies[-1], N )
    upconvPowerSpectrum = np.interp(linFreqs, frequencies, upconvPowerSpectrum)
    
    
    newMovingMirrorData = np.zeros((movingMirrorData.shape[0], linFreqs.shape[0]))
    
    for ii in range( movingMirrorData.shape[0] ):
        newMovingMirrorData[ii,:] = np.interp(linFreqs, frequencies, movingMirrorData[ii,:])
    
    movingMirrorData = newMovingMirrorData
    frequencies = linFreqs
    
    
    
    #crossCorr = np.zeros( (movingMirrorData.shape[0], movingMirrorData.shape[1]))
    crossCorr = np.zeros( (movingMirrorData.shape[0], movingMirrorData.shape[1]*2-1))

    lags = np.zeros_like(crossCorr)
    shearMap = np.zeros( movingMirrorData.shape[0] )
    
    for ii in range( movingMirrorData.shape[0] ):
        #lags[ii,:], crossCorr[ii,:] =fq.ezcorr(frequencies, movingMirrorData[ii,:], upconvPowerSpectrum) 
        #shearMap[ii] = lags[ii,:][ crossCorr[ii,:] == np.max(crossCorr[ii,:]) ]
        crossCorr[ii,:] =  np.correlate(movingMirrorData[ii,:], upconvPowerSpectrum,'full')
        maxId = np.argmax(crossCorr[ii,:])
        peakFreq = -(N - (maxId+1))*df
        lags = -(N - np.linspace(1,2*N-1,2*N-1) ) *df
        
        
        x,y = fq.ezdiff(lags, crossCorr[ii,:])
        
        f = interp.interp1d(x, y, kind = 'cubic')
        
        err = 1
        threshold = 1e-5
        maxIter = 1000
        nIter = 0
        x0 = peakFreq - 5*df
        x1 = peakFreq + 5*df
            
        while err > threshold:
            nIter += 1
            if nIter > maxIter:
                break
                
            if x0<np.min(x):
                x0 = np.min(x)
            if x0>np.max(x):
                x0 = np.max(x)
                
            if x1<np.min(x):
                x1 = np.min(x)
            if x1>np.max(x):
                x1 = np.max(x)
                    
            f0 = f(x0)
            f1 = f(x1)
            dfdx = (f1-f0) / (x1 - x0)
            b = f0 - dfdx*x0
            
            x0 = x1
            x1 = -b/dfdx
            err = abs((x1-x0)/x0)
            
             
        
        shearMap[ii] = x1# -(N - (maxId+1))*df
    
    p = np.polyfit(movingMirror_Z, shearMap, 1)
    
    shear = np.polyval(p, (tdsiTraceZ[0]+tdsiTraceZ[-1])/2)
        
    if debugGraphs:
        plt.figure()
        plt.plot(movingMirror_Z,shearMap/1e12, LineWidth = 2, label = 'Data')
        plt.plot(movingMirror_Z, np.polyval(p, movingMirror_Z) /1e12, 'r', LineWidth = 2, label = 'Linear fit')
        plt.legend()
        plt.xlabel(r'Stage displacement [$\mu$m]')
        plt.ylabel('Shear [THz]')
        
        plt.figure()
        plt.pcolor(lags/1e12, movingMirror_Z, crossCorr)
        plt.plot(shearMap/1e12, movingMirror_Z, '--r')
        plt.xlim(-20,20)
        plt.xlabel('Shear frequency [THz]')
        plt.ylabel(r'Stage displacement [$\mu$m]')
    
    return shear





def make1Dfft(wavelengths,stagePosition,trace,zeropadding = True, windowing = True, debugGraphs= False):

    #stagePosition = np.abs(stagePosition)
    
    dz = np.diff(stagePosition)[0]
    
    
    plt.figure()
    plt.pcolor(wavelengths*1e9,stagePosition,np.abs(trace)/np.max(np.abs(trace)))
    plt.xlabel(r'Wavelengths [nm]')
    plt.ylabel(r'Stage displacement [$\mu$m]')
    c = plt.colorbar()
    c.set_label('Relative intensity')
    
    
    if windowing:
        hanningWindow = np.hanning(stagePosition.shape[0])
        hanningWindow = np.transpose( np.tile(hanningWindow, (wavelengths.shape[0],1)))
        trace = trace * hanningWindow
        
    if zeropadding:
        deficit = 2**NextPowerOfTwo(stagePosition.shape[0]) - stagePosition.shape[0]
        deficitUp = int(np.floor(deficit/2))
        deficitDown = int(np.ceil(deficit/2))
        
        zeroPadUp = np.zeros((deficitUp,wavelengths.shape[0]))
        zeroPadDown = np.zeros((deficitDown,wavelengths.shape[0]))
        trace = np.concatenate((zeroPadUp, trace, zeroPadDown))
    
    
    
    # Perform 1D fft of 2DSI trace along stage position dimension
    fTrace = np.fft.fftshift(np.fft.fft(trace, axis = 0), axes = 0)
    
    # Calculate spatial frequencies axis
    spatialFreqs = np.fft.fftshift( np.fft.fftfreq(fTrace.shape[0], dz) )
    
    # Keep only positive frequencies
    iiMid = spatialFreqs>=0
    fTrace = fTrace[iiMid,:]
    spatialFreqs = spatialFreqs[iiMid]
    
    # Take absolute value of interferogram and find peak corresponding to spectral
    # phase oscillations
    absfTrace = np.abs( fTrace )
    
    
    tmp = np.mean(absfTrace, axis = 1)
    II = sig.find_peaks(tmp)[0]
    II = II[tmp[II] == np.max(tmp[II])]
    
    phase = np.squeeze( np.unwrap( np.angle( fTrace[II,:] ) ) )
    amplitude = np.squeeze( absfTrace[II,:] )
    spatialFreq = spatialFreqs[II]
    
    if debugGraphs:
        plt.figure()
        plt.pcolor(wavelengths*1e9, spatialFreqs, absfTrace/np.max(absfTrace))
        plt.xlabel('Wavelengths [nm]')
        plt.ylabel(r'Spatial frequency [$\mu$m$^{-1}$]')
        c = plt.colorbar()
        c.set_label('Relative intensity')
    
    return spatialFreq, amplitude, phase
               
               
def cut_fft(amplitude, phase, wavelengths,debugGraphs):
    
    cut = 0.01
    extend = 0
    
    Icut = amplitude >= cut*np.max(amplitude)
    Imin = np.argwhere(Icut)[0]
    Imax = np.argwhere(Icut)[-1]
    Idelta = (Imax - Imin)*extend
    
    Imin = np.squeeze( np.ceil(Imin - Idelta).astype(int) )
    Imax = np.squeeze( np.floor(Imax + Idelta).astype(int) )
    
    Icut = np.zeros_like(Icut, dtype = int)
    Icut[Imin : Imax+1] = 1
    IcutBool = Icut.astype(bool)
    #IcutIndex = np.squeeze( np.argwhere(Icut) )
    
    if debugGraphs:
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(wavelengths*1e9, phase, label = 'Full data')
        
    amplitude = amplitude[IcutBool]
    phase = phase[IcutBool]
    FFTwavelengths = wavelengths[IcutBool]
    
               
    amplitude = amplitude / np.max(amplitude)
               
    
    phase = phase - np.fix( np.min( phase/(2*np.pi) ) )*2*np.pi
        
    if debugGraphs:       
        ax.plot(FFTwavelengths*1e9, phase,'--', label = 'Trunc. data')
        ax.set_xlabel('Wavelength [nm]')
        ax.set_ylabel('2DSI fringes phase [rad]')
        ax.legend()
    return amplitude, phase, FFTwavelengths
               
               
               

def calc_spectral_phase(shearFrequency, upconvWavelength,FFTamplitude, FFTphase, FFTwavelengths,polyfit,debugGraphs):
    
    
    shearW = 2*np.pi * shearFrequency
    
    phase  = -FFTphase # Minus to "correct GVD sign"...
    
    concW, concPhase,concGD, concGDD, concTOD = phase_concatenation(FFTwavelengths,FFTamplitude, shearW, phase, upconvWavelength,polyfit=polyfit ,debugGraphs =debugGraphs)
    midpointW, midpointPhase, midpointGD, midpointGDD, midpointTOD = phase_midpoint(FFTwavelengths,FFTamplitude, shearW, phase, upconvWavelength,polyfit=polyfit ,debugGraphs =debugGraphs)
    
    return concW, concPhase,concGD, concGDD, concTOD, midpointW, midpointPhase,midpointGD, midpointGDD, midpointTOD
    
    
    
def phase_concatenation(wavelengths,amplitude, shearW, phase, upconvWavelength, polyfit = 0,debugGraphs= False):
    
    wCut = 2*np.pi * C *( 1/wavelengths - 1/upconvWavelength)
    
    dwCut = np.mean( np.diff( wCut ) )
    
    shearMultiplicity = np.round( np.abs( shearW / dwCut ) ).astype(int)
    
    newDW = np.abs( shearW / shearMultiplicity)
    
    N = np.floor( (np.max(wCut) - np.min(wCut)) / newDW ).astype(int)
    
    wMin = np.max(wCut) - N*newDW
    
    wCutLin = np.linspace( wMin , np.max(wCut), N+1 )
    
    phaseCutLin = np.interp(wCutLin, np.flip(wCut), np.flip(phase))
    amplitudeCutLin = np.interp(wCutLin, np.flip(wCut), np.flip(amplitude))
    
    
    concPhase = np.zeros((shearMultiplicity, np.floor(phaseCutLin.shape[0]/shearMultiplicity).astype(int)+1 ))
    concW = np.zeros((shearMultiplicity, np.floor(phaseCutLin.shape[0]/shearMultiplicity).astype(int) ))
    
    
    for ii in range(np.floor(phaseCutLin.shape[0]/shearMultiplicity).astype(int)):
        
        concPhase[:, ii+1] = (concPhase[:, ii].transpose() - phaseCutLin[(ii)*shearMultiplicity : (ii+1)*shearMultiplicity]).transpose()
        concW[:, ii]= wCutLin[(ii)*shearMultiplicity : (ii+1)*shearMultiplicity].transpose()
    
    concPhase = concPhase[:,1:]
    
    if shearW < 0:
        concPhase = -concPhase
    
    concGD = np.zeros_like(concPhase[:,1:-1])
    concGDD = np.zeros_like(concPhase[:,1:-1])
    concTOD = np.zeros_like(concPhase[:,2:-2])
    
    for ii in range(shearMultiplicity):
        
        
        weights = np.interp(concW[ii,:], wCutLin, amplitudeCutLin)
        linFit = np.polyfit(concW[ii,:], concPhase[ii,:], 1, w = weights)
        
        concPhase[ii,:] = concPhase[ii,:] - np.polyval(linFit, concW[ii,:])
    
        if polyfit >= 2:
            phaseFit =  np.polyfit(concW[ii,:], concPhase[ii,:], polyfit, w = weights)
            concPhase[ii,:] = np.polyval(phaseFit, concW[ii,:])
        
        concGD[ii,:] = fq.ezdiff(concW[ii,:], concPhase[ii,:], n=1, order = 2)[1]
        concGDD[ii,:] = fq.ezdiff(concW[ii,:], concPhase[ii,:], n=2, order = 2)[1]
        concTOD[ii,:] = fq.ezdiff(concW[ii,:], concPhase[ii,:], n=3, order = 2)[1]
    
    if debugGraphs:
        plt.figure()
        plt.plot((2*np.pi*C/concW.transpose()*1e9), concPhase.transpose())
        plt.ylabel('Spectral phase (concat. method) [rad]')
        plt.xlabel('Downconv. wavelength [nm]')
        
    return concW, concPhase, concGD, concGDD, concTOD
        

def phase_midpoint(wavelengths,amplitude, shearW, phase, upconvWavelength, polyfit = 0,debugGraphs= False):
    
    wCut = 2*np.pi * C *( 1/wavelengths - 1/upconvWavelength)
    
    dwCut = np.mean( np.diff( wCut ) )
    
    shearMultiplicity = np.round( np.abs( shearW / dwCut ) ).astype(int)
    
    newDW = np.abs( shearW / shearMultiplicity)
    
    N = np.floor( (np.max(wCut) - np.min(wCut)) / newDW ).astype(int)
    
    wMin = np.max(wCut) - N*newDW
    
    wCutLin = np.linspace( wMin , np.max(wCut), N+1 )
    
    phaseCutLin = np.interp(wCutLin, np.flip(wCut), np.flip(phase))
    amplitudeCutLin = np.interp(wCutLin, np.flip(wCut), np.flip(amplitude))
    
    
    
    deltaOfW = integrate.cumtrapz(phaseCutLin, wCutLin)
    midpointW = wCutLin[1:]
    
    midpointPhase = -np.interp(midpointW-shearW/2, midpointW, deltaOfW) / shearW
   
    
    
    
    
    midpointGD = np.zeros_like(midpointPhase[1:-1])
    midpointGDD = np.zeros_like(midpointPhase[1:-1])
    midpointTOD = np.zeros_like(midpointPhase[2:-2])
    
    weights = np.interp(midpointW, wCutLin, amplitudeCutLin)
    linFit = np.polyfit(midpointW, midpointPhase, 1, w = weights)
        
    midpointPhase = midpointPhase - np.polyval(linFit, midpointW)
    
    if polyfit >= 2:
        phaseFit =  np.polyfit(midpointW, midpointPhase, polyfit, w = weights)
        midpointPhase = np.polyval(phaseFit, midpointW)
    
    
    midpointGD = fq.ezdiff(midpointW, midpointPhase, n=1, order = 2)[1]
    midpointGDD = fq.ezdiff(midpointW, midpointPhase, n=2, order = 2)[1]
    midpointTOD = fq.ezdiff(midpointW, midpointPhase, n=3, order = 2)[1]
    
    if debugGraphs:
        plt.figure()
        plt.plot((2*np.pi*C/midpointW.transpose()*1e9), midpointPhase.transpose())
        plt.ylabel('Spectral phase (midpoint method) [rad]')
        plt.xlabel('Downconv. wavelength [nm]')
        
    return midpointW, midpointPhase,midpointGD, midpointGDD, midpointTOD


def calc_temporal_envelope(wavelengths, upconvPowerSpectrum, upconvWavelength, angFreqPhase, phase, plotSpectrum,minTimeResolution):
    
    pulse = []
    
    if len(phase.shape) > 1:
    
        for ii in range(phase.shape[0]):
        
            frequency = np.flip( C * (1/wavelengths - 1/upconvWavelength) )
            
            spectrumEnvelope = np.flip( np.sqrt(upconvPowerSpectrum/np.max(upconvPowerSpectrum)) )
            
            nuMax = np.max(frequency)
            nuMin = np.min(frequency)
            dNu = np.min( np.diff( frequency ) )
            
            if minTimeResolution < 1/(2*nuMax):
                newNuMax = 1 / (2* minTimeResolution)
                nu = np.linspace(-newNuMax, newNuMax, int(np.floor(2*newNuMax/dNu))+1)
            else:
                nu = np.linspace(-nuMax, nuMax, int(np.floor(2*nuMax/dNu))+1)
            
            interpEnvelope = np.zeros_like(nu)
            interpPhase = np.zeros_like(nu)
            
            II = ( (nu>=nuMin) & (nu<=nuMax) )
            
            interpEnvelope[II] = np.interp(nu[II], frequency, spectrumEnvelope)
            interpPhase[II] = np.interp(nu[II], angFreqPhase[ii,:]/(2*np.pi), phase[ii,:])
            interpPhase[nu<nuMin] = interpPhase[II][0]
            interpPhase[nu>nuMax] = interpPhase[II][-1]
            
            spectrum = interpEnvelope * np.exp(1j * interpPhase)
            
            t, eField = fq.ezifft(nu, spectrum)
            eField = np.fft.ifftshift(eField)
            
            pulse.append( np.abs(eField) )
# =============================================================================
#             if (plotSpectrum and ii==0):
#                 IIplotphase = interpEnvelope[nu>0] > np.max(interpEnvelope)/10
#                 
#                 
#                 fig = plt.figure()
#                 axLeft = fig.gca()
#                 axRight = axLeft.twinx()
#                 axLeft.plot(C/nu[nu>0]*1e9, interpEnvelope[nu>0]**2)
#                 axRight.plot(C/nu[nu>0][IIplotphase]*1e9,interpPhase[nu>0][IIplotphase], '--r')
#                 axLeft.set_xlim(np.min(C/nu[nu>0]*1e9),1200)
#                 axLeft.set_xlabel('Wavelength [nm]')
#                 axLeft.set_ylabel('Normalized intensity', color = 'blue')
#                 axRight.set_ylabel('Spectral phase [rad]', color = 'red')
# =============================================================================
    
    else:
        frequency = np.flip( C * (1/wavelengths - 1/upconvWavelength) )
            
        spectrumEnvelope = np.flip( np.sqrt(upconvPowerSpectrum/np.max(upconvPowerSpectrum)) )
        
        nuMax = np.max(frequency)
        nuMin = np.min(frequency)
        dNu = np.min( np.diff( frequency ) )
        
        if minTimeResolution < 1/(2*nuMax):
                newNuMax = 1 / (2* minTimeResolution)
                nu = np.linspace(-newNuMax, newNuMax, int(np.floor(2*newNuMax/dNu))+1)
        else:
                nu = np.linspace(-nuMax, nuMax, int(np.floor(2*nuMax/dNu))+1)
            
        interpEnvelope = np.zeros_like(nu)
        interpPhase = np.zeros_like(nu)
        
        II = ( (nu>=nuMin) & (nu<=nuMax) )
        
        interpEnvelope[II] = np.interp(nu[II], frequency, spectrumEnvelope)
        interpPhase[II] = np.interp(nu[II], angFreqPhase/(2*np.pi), phase)
        interpPhase[nu<nuMin] = interpPhase[II][0]
        interpPhase[nu>nuMax] = interpPhase[II][-1]
        
        spectrum = interpEnvelope * np.exp(1j * interpPhase)
        
        t, eField = fq.ezifft(nu, spectrum)
        eField = np.fft.ifftshift(eField)
        
        pulse.append( np.abs(eField) )
# =============================================================================
#         if (plotSpectrum):
#             IIplotphase = interpEnvelope[nu>0] > np.max(interpEnvelope)/10
#             
#             
#             fig = plt.figure()
#             axLeft = fig.gca()
#             axRight = axLeft.twinx()
#             axLeft.plot(C/nu[nu>0]*1e9, interpEnvelope[nu>0]**2)
#             axRight.plot(C/nu[nu>0][IIplotphase]*1e9,interpPhase[nu>0][IIplotphase], '--r')
#             axLeft.set_xlim(np.min(C/nu[nu>0]*1e9),1200)
#             axLeft.set_xlabel('Wavelength [nm]')
#             axLeft.set_ylabel('Normalized intensity', color = 'blue')
#             axRight.set_ylabel('Spectral phase [rad]', color = 'red')
# =============================================================================
        
    
    pulse = np.array(pulse)
    pulse = np.mean(pulse, axis = 0)
    pulse = pulse**2 / np.max(pulse**2)
    
    
    
    return t, pulse, eField
    



def make_shear_map(wavelengths, upconvPowerSpectrum, movingMirrorData, movingMirror_Z,debugGraphs):
    """ Calculate shear frequency as a function of stage position,
    and take its value at the middle position as constant approximation """
    
    # Normalize spectra to max of one
    upconvPowerSpectrum = upconvPowerSpectrum / np.max(upconvPowerSpectrum)
    
    maxValues = np.max(movingMirrorData, axis = 1)
    tmp = wavelengths.shape[0]
    maxValues = np.transpose( np.tile(maxValues, (tmp,1)) )
    movingMirrorData = movingMirrorData / maxValues
    


    # Convert wavelengths to frequencies
    frequencies = C / wavelengths
    
    frequencies = np.flip(frequencies) # Flipping from low to high frequencies
    upconvPowerSpectrum = np.flip(upconvPowerSpectrum)
    movingMirrorData = np.flip(movingMirrorData,axis = 1)
    
    # Interpolate to a linear spacing of frequencies
    # Choice of datapoint position strongly affect results, here I am copying Matlab
    # need to check if another strategy would work better
    Df = frequencies[-1] - frequencies[0]
    df = np.max( np.diff(frequencies) ) / 2 #Added this division by two as a test
    N = round(Df / df)
    linFreqs = np.linspace(frequencies[-1]-(N-1)*df, frequencies[-1], N )
    upconvPowerSpectrum = np.interp(linFreqs, frequencies, upconvPowerSpectrum)
    
    
    newMovingMirrorData = np.zeros((movingMirrorData.shape[0], linFreqs.shape[0]))
    
    for ii in range( movingMirrorData.shape[0] ):
        newMovingMirrorData[ii,:] = np.interp(linFreqs, frequencies, movingMirrorData[ii,:])
    
    movingMirrorData = newMovingMirrorData
    frequencies = linFreqs
    
    
    
    #crossCorr = np.zeros( (movingMirrorData.shape[0], movingMirrorData.shape[1]))
    crossCorr = np.zeros( (movingMirrorData.shape[0], movingMirrorData.shape[1]*2-1))

    lags = np.zeros_like(crossCorr)
    shearMap = np.zeros( movingMirrorData.shape[0] )
    
    for ii in range( movingMirrorData.shape[0] ):
        #lags[ii,:], crossCorr[ii,:] =fq.ezcorr(frequencies, movingMirrorData[ii,:], upconvPowerSpectrum) 
        #shearMap[ii] = lags[ii,:][ crossCorr[ii,:] == np.max(crossCorr[ii,:]) ]
        crossCorr[ii,:] =  np.correlate(movingMirrorData[ii,:], upconvPowerSpectrum,'full')
        maxId = np.argmax(crossCorr[ii,:])
        
        lags = -(N - np.linspace(1,2*N-1,2*N-1) ) *df
        peakFreq = np.average(lags, weights = crossCorr[ii,:])
        
        
        shearMap[ii] = peakFreq#-(N - (maxId+1))*df
    
    p = np.polyfit(movingMirror_Z, shearMap, 1)
    
    shearMap = np.polyval(p, movingMirror_Z)
        
    if debugGraphs:
        plt.figure()
        plt.plot(movingMirror_Z/1000,shearMap/1e12, LineWidth = 2, label = 'Data')
        plt.plot(movingMirror_Z/1000, np.polyval(p, movingMirror_Z) /1e12, 'r', LineWidth = 2, label = 'Linear fit')
        plt.legend()
        plt.xlabel(r'Stage displacement [$\mu$m]')
        plt.ylabel('Shear [THz]')
    
    return shearMap


def simFrogTrace(t,E,minWavelength,maxWavelength):
    
    I = np.abs(E)**2
    
    t0 = -200e-15#t[t<0][I[t<0]<np.max(I)/100][-1]*2
    t1 = 200e-15#t[t>0][I[t>0]<np.max(I)/100][0]*2
    
    if abs(t1) > abs(t0):
        t0 = -t1
    else:
        t1 = -t0
    
    lags = np.linspace(t0, t1, 400)
    
    wavelengths = np.linspace(minWavelength, maxWavelength, 1000)
    frogTrace = np.zeros((wavelengths.shape[0],lags.shape[0]))
    
    for ii,lag in enumerate(lags):
        
        t2 = t + lag
        E2 = np.interp(t,t2,E)
        
        Esfg = E * E2
        
        
        nu, S = fq.ezfft(t, Esfg)
        
        wav = C/np.flip(nu)
        S = np.flip(S)
        spectrum = np.interp(wavelengths, wav, np.abs(S)**2)
        
        frogTrace[:,ii] = spectrum
    
    frogTrace = frogTrace / np.max(frogTrace)
    
    plt.figure()
    plt.pcolor(lags*1e15, wavelengths*1e9, frogTrace, cmap = 'viridis')
    plt.gca().invert_yaxis()
    plt.xlabel('Delay [fs]')
    plt.ylabel('Wavelength [nm]')
    c = plt.colorbar()
    c.set_label('Relative intensity')
    
    
        
        
    
    