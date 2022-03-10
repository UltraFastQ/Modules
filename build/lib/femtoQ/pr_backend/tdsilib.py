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


def substractNoiseFloor(fixedMirrorData, movingMirrorData,relativeNoiseFloor):
    """ Substract noise floor from spectra (except trace) """
    
    lowLimit = relativeNoiseFloor
    
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
    upconvPowerSpectrum = np.interp(linFreqs, frequencies, upconvPowerSpectrum*C/frequencies**2)
    
    
    newMovingMirrorData = np.zeros((movingMirrorData.shape[0], linFreqs.shape[0]))
    
    for ii in range( movingMirrorData.shape[0] ):
        newMovingMirrorData[ii,:] = np.interp(linFreqs, frequencies, movingMirrorData[ii,:]*C/frequencies**2)
    
    movingMirrorData = newMovingMirrorData
    frequencies = linFreqs
    
    shearMap = np.zeros( movingMirrorData.shape[0] )
    v0 = np.trapz(upconvPowerSpectrum*linFreqs,linFreqs) / np.trapz(upconvPowerSpectrum,linFreqs)
    
    for ii in range( movingMirrorData.shape[0] ):
        shearMap[ii] = np.trapz(movingMirrorData[ii,:]*linFreqs,linFreqs) / np.trapz(movingMirrorData[ii,:],linFreqs) - v0

    
    p = np.polyfit(movingMirror_Z, shearMap, 1)
    
    shear = np.polyval(p, (tdsiTraceZ[0]+tdsiTraceZ[-1])/2)
        
    if debugGraphs:
        plt.figure()
        plt.plot(movingMirror_Z,shearMap/1e12,'b', linewidth = 2, label = 'Data')
        plt.plot(movingMirror_Z, np.polyval(p, movingMirror_Z) /1e12, '--r', linewidth = 2, label = 'Linear fit')
        plt.legend()
        plt.xlabel(r'Stage displacement [$\mu$m]')
        plt.ylabel('Shear [THz]')
        
    
    return shear


def find_upconversion(wavelengths, upconvPowerSpectrum, fundWavelengths, fundPowerSpectrum):
    """ Calculate shear frequency as a function of stage position,
    and take its value at the middle position as constant approximation """
    
    # Normalize spectra to max of one
    upconvPowerSpectrum = upconvPowerSpectrum / np.max(upconvPowerSpectrum)
    
    fundPowerSpectrum = fundPowerSpectrum/fundPowerSpectrum.max()


    # Convert wavelengths to frequencies
    frequencies = C / wavelengths
    fundFrequencies = C/ fundWavelengths
    
    frequencies = np.flip(frequencies) # Flipping from low to high frequencies
    fundFrequencies = np.flip(fundFrequencies)
    upconvPowerSpectrum = np.flip(upconvPowerSpectrum)
    fundPowerSpectrum = np.flip(fundPowerSpectrum)
    
    # Interpolate to a linear spacing of frequencies
    # Choice of datapoint position strongly affect results, here I am copying Matlab
    # need to check if another strategy would work better
    Df = np.min( [ np.min(np.diff(frequencies)),  np.min(np.diff(fundFrequencies))  ]  )
    maxFreq = frequencies.max()
    minFreq = fundFrequencies.min()
    
    N = int(round((maxFreq -minFreq) / Df))
    linFreqs = np.linspace(maxFreq, minFreq, N )
    
    upconvPowerSpectrum = np.interp(linFreqs, frequencies, upconvPowerSpectrum*C/frequencies**2)
    fundPowerSpectrum = np.interp(linFreqs, fundFrequencies, fundPowerSpectrum*C/fundFrequencies**2)
    
    frequencies = linFreqs
    
    v0Fund = np.trapz(fundPowerSpectrum*linFreqs,linFreqs) / np.trapz(fundPowerSpectrum,linFreqs)
    v0Upconv = np.trapz(upconvPowerSpectrum*linFreqs,linFreqs) / np.trapz(upconvPowerSpectrum,linFreqs)
    
    upconvFrequency = v0Upconv - v0Fund
    upconvWavelength = C / upconvFrequency
    
    newWav = C/(linFreqs-upconvFrequency)
    
    downConvWavelengths = np.linspace(fundWavelengths.min(),fundWavelengths.max(),fundWavelengths.shape[0])
    downConvSpectrum =  np.interp(fundWavelengths,newWav[newWav>0] , upconvPowerSpectrum[newWav>0]*(linFreqs-upconvFrequency)[newWav>0]**2/C)
    
    
    return upconvWavelength, downConvWavelengths, downConvSpectrum

def make1Dfft(wavelengths,stagePosition,trace,zeropadding = True, windowing = True, debugGraphs= False):

    #stagePosition = np.abs(stagePosition)
    
    dz = np.diff(stagePosition)[0]
    
    zMin = stagePosition[0]
    zMax = stagePosition[-1]
    
    plt.figure()
    plt.pcolormesh(wavelengths*1e9,(stagePosition-(zMin+zMax)/2)*1e-6*2/C*1e15,np.abs(trace)/np.max(np.abs(trace)),shading='auto')
    plt.xlabel(r'Wavelengths [nm]')
    plt.ylabel(r'Delay [fs]')
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
        plt.pcolormesh(wavelengths*1e9, spatialFreqs/1e-6/2*C/1e12, absfTrace/np.max(absfTrace),shading='auto')
        plt.xlabel('Wavelengths [nm]')
        plt.ylabel(r'Frequency [THz]')
        c = plt.colorbar()
        c.set_label('Relative intensity')
    
    return spatialFreq, amplitude, phase
               
               
def cut_fft(amplitude, phase, wavelengths,debugGraphs):
    
    cut = 0.00
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
               
               
               

def calc_spectral_phase(shearFrequency, upconvWavelength,FFTamplitude, FFTphase, FFTwavelengths,debugGraphs):
    
    
    shearW = 2*np.pi * shearFrequency
    
    phase  = -FFTphase # Minus to "correct GVD sign"...
    
    concW, concPhase = phase_concatenation(FFTwavelengths,FFTamplitude, shearW, phase, upconvWavelength ,debugGraphs =debugGraphs)
    midpointW, midpointPhase = phase_midpoint(FFTwavelengths,FFTamplitude, shearW, phase, upconvWavelength ,debugGraphs =debugGraphs)
    
    return concW, concPhase, midpointW, midpointPhase
    
    
    
def phase_concatenation(wavelengths,amplitude, shearW, phase, upconvWavelength,debugGraphs= False):
    
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
    
    
    for ii in range(shearMultiplicity):
        
        
        weights = np.interp(concW[ii,:], wCutLin, amplitudeCutLin)
        linFit = np.polyfit(concW[ii,:], concPhase[ii,:], 1, w = weights)
        
        concPhase[ii,:] = concPhase[ii,:] - np.polyval(linFit, concW[ii,:])
    
    
    if debugGraphs:
        plt.figure()
        plt.plot((2*np.pi*C/concW.transpose()*1e9), concPhase.transpose())
        plt.ylabel('Spectral phase (concat. method) [rad]')
        plt.xlabel('Downconv. wavelength [nm]')
        
    return concW, concPhase
        

def phase_midpoint(wavelengths,amplitude, shearW, phase, upconvWavelength,debugGraphs= False):
    
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
    
    midpointPhase = -interp.interp1d(midpointW, deltaOfW,'linear',bounds_error = False, fill_value = 'extrapolate')(midpointW+shearW/2) / shearW
   
    
    weights = np.interp(midpointW, wCutLin, amplitudeCutLin)
    linFit = np.polyfit(midpointW, midpointPhase, 1, w = weights)
        
    midpointPhase = midpointPhase - np.polyval(linFit, midpointW)
    
    
    if debugGraphs:
        plt.figure()
        plt.plot((2*np.pi*C/midpointW.transpose()*1e9), midpointPhase.transpose())
        plt.ylabel('Spectral phase (midpoint method) [rad]')
        plt.xlabel('Downconv. wavelength [nm]')
        
    return midpointW, midpointPhase


def calc_temporal_envelope(wavelengths, upconvPowerSpectrum, upconvWavelength, angFreqPhase, phase, plotSpectrum,minTimeResolution):
    
    pulse = []
    eFieldOut = []
    
    frequency = np.flip( C * (1/wavelengths - 1/upconvWavelength) )
    spectrumEnvelope = np.flip(  upconvPowerSpectrum* wavelengths**2/C/np.max(upconvPowerSpectrum* wavelengths**2/C)  )
    
    if len(phase.shape) > 1:
    
        for ii in range(phase.shape[0]):
        
            
            
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
            
            interpEnvelope[II] = np.sqrt( np.interp(nu[II], frequency, spectrumEnvelope) )
            interpPhase[II] = np.interp(nu[II], angFreqPhase[ii,:]/(2*np.pi), phase[ii,:])
            interpPhase[nu<nuMin] = interpPhase[II][0]
            interpPhase[nu>nuMax] = interpPhase[II][-1]
            
            spectrum = interpEnvelope * np.exp(1j * interpPhase)
            
            t, eField = fq.ezifft(nu, spectrum, amplitudeSpectrumRecentering = True)
            
            eFieldOut.append( eField )
            pulse.append( np.abs(eField) )
    
    else:
            
        
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
        
        interpEnvelope[II] =np.sqrt(  np.interp(nu[II], frequency, spectrumEnvelope) )
        interpPhase[II] = np.interp(nu[II], angFreqPhase/(2*np.pi), phase)
        interpPhase[nu<nuMin] = interpPhase[II][0]
        interpPhase[nu>nuMax] = interpPhase[II][-1]
        
        spectrum = interpEnvelope * np.exp(1j * interpPhase)
        
        t, eField = fq.ezifft(nu, spectrum, amplitudeSpectrumRecentering = True)
        
        eFieldOut.append( eField )
        pulse.append( np.abs(eField) )
        
    
    eFieldOut = np.array(eFieldOut)
    eFieldOut = np.mean(eFieldOut, axis = 0)
    
    pulse = np.array(pulse)
    pulse = np.mean(pulse, axis = 0)
    pulse = pulse**2 / np.max(pulse**2)
    
    
    
    return t, pulse, eFieldOut
    




    