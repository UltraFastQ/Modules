# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 16:18:05 2021

@author: cusso
"""

from tools import *
import numpy as np
import matplotlib.pyplot as plt
import simulate_pulse_retrieval as spr
import pulse_retrieval as pr

# =============================================================================
# N = 2**17
# T = 3e-12
# 
# v0 = 50e12
# tau = 260e-15
# 
# w0 = 2*np.pi*v0
# dwii = 2*np.pi*10e12
# 
# t = np.linspace(-T/2,T/2,N)
# dt = t[1]-t[0]
# Tactual = t[-1] - t[0]
# 
# E = np.exp(1j*w0*t) * np.exp(-2*t**2/tau**2)
# 
# 
# v,s = ezfft(t,E,backend = 'mkl',axis = 0)
# 
# t2,E2 = ezifft(v,s,backend = 'numpy',axis = 0)
# t3,E3 = ezifft(v,np.abs(s),backend = 'pyfftw',amplitudeSpectrumRecentering = True,axis = 0)
# 
# p1 = np.trapz(np.abs(E)**2,t,axis = 0)
# p2 = np.trapz(np.abs(s)**2,v,axis = 0)
# p3 = np.trapz(np.abs(E2)**2,t2,axis = 0)
# p4 = np.trapz(np.abs(E3)**2,t3,axis = 0)
# 
# plt.figure()
# plt.plot(t,E.real)
# plt.plot(t2,E2.real,'--')
# plt.plot(t3,E3.real,'--')
# 
# plt.figure()
# plt.plot(t,E.real-E2.real)
# plt.plot(t,E.real-E3.real)
# =============================================================================

# =============================================================================
# L = 10e-2
# a = 1e-5
# 
# dx = 100e-9 * L / a
# 
# x = np.linspace(dx, 1100e-9 * L / a, 2048)
# 
# y = a*x/np.sqrt(x**2+L**2)
# 
# p = np.exp(-2*(y-630e-9)**2/20e-9**2) + np.exp(-2*(y-750e-9)**2/100e-9**2)
# 
# 
# plt.figure()
# plt.plot(y,p)
# 
# 
# v, a = powerSpectrum2fftAmplitudeSpectrum(y,p,'', 'next')
# 
# t,E = ezifft(v,a,-1,True)
# 
# plt.figure()
# plt.plot(v/1e12,a)
# 
# plt.figure()
# plt.plot(t,np.abs(E)**2)
# =============================================================================

# =============================================================================
# x = np.linspace(-100,100,10001)
# n = (np.random.rand(10001)*2 - 1)*0.1
# 
# f = np.exp(-2*x**2/10**2)
# 
# y = f+n
# 
# 
# z = ezpad(x,f,10,100)
# 
# =============================================================================


# =============================================================================
# spr.TwoDSI()
# t, Ec, Em = pr.twodsi('simulated2dsiData.npz',smoothSpectrum = True,relativeNoiseFloor = 0.01,spectrum = 'simulated2dsiFundSpectrum.npz',simulatedData = True,debug=True)
# =============================================================================

# =============================================================================
# spr.FROG()
# pr.shgFROG('simFrogData.npz', initialGuess = 'gaussian', smoothTrace = False, relativeNoiseTreshold=0.00,maxIter = 100,marginalCorrection = 'simFROGspectrum.npz')
# =============================================================================

# =============================================================================
# data = np.load('pumpSpectrum.npz')
# wav = data['wavelengths']*1e-9
# S = data['spectrum']
# 
# data.close()
# 
# v,s = powerSpectrum2fftAmplitudeSpectrum(wav, S, frequencySpacing = 'average',powerOfTwo = 'next')
# 
# t,E = ezifft(v,s,amplitudeSpectrumRecentering = True)
# 
# 
# 
# tau = ezfindwidth(t,np.abs(E)**2)
# 
# plt.figure()
# plt.plot(t,np.abs(E)**2)
# =============================================================================
