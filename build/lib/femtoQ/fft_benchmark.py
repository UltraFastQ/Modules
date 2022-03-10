# -*- coding: utf-8 -*-

import timeit
import numpy as np
import matplotlib.pyplot as plt

#%% Parameters
N = 100             # Number of iterations for averaging
L = 10              # Nummber of ffts to try in parallel for mkl and cupy
includeCupy = False # If false, cupy calculations will be skipped
pwrMin = 10          # Min power of 2 vector length to test
pwrMax = 16         # Max power of 2 vector length to test


#%% Calculations

pwrOf2 = np.linspace(pwrMin,pwrMax,pwrMax-pwrMin+1)

num = np.zeros_like(pwrOf2)
sci = np.zeros_like(pwrOf2)
fftw1 = np.zeros_like(pwrOf2)
fftw2 = np.zeros_like(pwrOf2)
fftw4 = np.zeros_like(pwrOf2)
fftw8 = np.zeros_like(pwrOf2)
fftw16 = np.zeros_like(pwrOf2)
mkl = np.zeros_like(pwrOf2)
cupy = np.zeros_like(pwrOf2)
mkl_par = np.zeros_like(pwrOf2)
cupy_par = np.zeros_like(pwrOf2)
ez_num = np.zeros_like(pwrOf2)
ez_sci = np.zeros_like(pwrOf2)
ez_fftw = np.zeros_like(pwrOf2)
ez_mkl = np.zeros_like(pwrOf2)
ez_cup = np.zeros_like(pwrOf2)

for ii, pwr in enumerate(pwrOf2):

    mysetup = '''
import numpy as np
import pyfftw
import scipy.fft
import mkl_fft
from tools import ezfft
pyfftw.interfaces.cache.enable()
a = np.random.rand(''' + str(int(round(2**pwr))) + ''').astype('complex128')
a_alt = np.random.rand(''' + str(int(round(2**pwr))) + ''','''+str(L)+''').astype('complex128')
t = np.linspace(0,1,a.shape[0])
b2 = mkl_fft.fft(a,axis = 0)
b2_alt = mkl_fft.fft(a_alt,axis = 0)
'''
    if includeCupy:
        mysetup += '''
import cupy as cp
acp = cp.array(a)
acp_alt = cp.array(a_alt)
b = cp.fft.fft(acp,axis = 0)
b_alt = cp.fft.fft(acp_alt,axis = 0)
'''
    
    
    num[ii] = timeit.timeit(setup = mysetup, stmt='b1 = np.fft.fft(a,axis = 0)',number = N)/N
    sci[ii] = timeit.timeit(setup = mysetup, stmt='b1 = scipy.fft.fft(a,axis = 0)',number = N)/N
    fftw1[ii] = timeit.timeit(setup = mysetup, stmt='b1 = pyfftw.interfaces.scipy_fftpack.fft(a, threads = 1,axis = 0)',number = N)/N
    #fftw2[ii] = timeit.timeit(setup = mysetup, stmt='b1 = pyfftw.interfaces.scipy_fftpack.fft(a, threads = 2,axis = 0)',number = N)/N
    #fftw4[ii] = timeit.timeit(setup = mysetup, stmt='b1 = pyfftw.interfaces.scipy_fftpack.fft(a, threads = 4,axis = 0)',number = N)/N
    #fftw8[ii] = timeit.timeit(setup = mysetup, stmt='b1 = pyfftw.interfaces.scipy_fftpack.fft(a, threads = 8,axis = 0)',number = N)/N
    fftw16[ii] = timeit.timeit(setup = mysetup, stmt='b1 = pyfftw.interfaces.scipy_fftpack.fft(a, threads = 16,axis = 0)',number = N)/N
    mkl[ii] = timeit.timeit(setup = mysetup, stmt= 'b1 = mkl_fft.fft(a,axis = 0)',number = N)/N
    mkl_par[ii] = timeit.timeit(setup = mysetup, stmt= 'b1 = mkl_fft.fft(a_alt,axis = 0)',number = N)/N/L
    ez_num[ii] = timeit.timeit(setup = mysetup, stmt='b1 = ezfft(t,a,axis = 0,backend = "numpy")',number = N)/N
    ez_sci[ii] = timeit.timeit(setup = mysetup, stmt='b1 = ezfft(t,a,axis = 0,backend = "scipy")',number = N)/N
    ez_fftw[ii] = timeit.timeit(setup = mysetup, stmt='b1 = ezfft(t,a,axis = 0,backend = "fftw")',number = N)/N
    ez_mkl[ii] = timeit.timeit(setup = mysetup, stmt='b1 = ezfft(t,a,axis = 0,backend = "mkl")',number = N)/N
    ez_cup[ii] = timeit.timeit(setup = mysetup, stmt='b1 = ezfft(t,a,axis = 0,backend = "cupy")',number = N)/N
    if includeCupy:
        cupy[ii] = timeit.timeit(setup = mysetup, stmt= 'b1 = cp.fft.fft(acp,axis = 0)',number = N)/N
        cupy_par[ii] = timeit.timeit(setup = mysetup, stmt= 'b1 = cp.fft.fft(acp_alt,axis = 0)',number = N)/N/L
    
#%% Plotting
fig = plt.figure()
ax = plt.gca()
ax.scatter(2**pwrOf2, num*1e6, label = 'numpy.fft.fft()',alpha = 1)
ax.scatter(2**pwrOf2, sci*1e6, label = 'scipy.fft.fft()',alpha = 1)
ax.scatter(2**pwrOf2, fftw1*1e6, label = 'pyFFTW interface - 1 thread',alpha = 1)
#ax.scatter(2**pwrOf2, fftw2*1e6, label = 'pyFFTW interface - 2 threads',alpha = 0.5)
#ax.scatter(2**pwrOf2, fftw4*1e6, label = 'pyFFTW interface - 4 threads',alpha = 0.5)
#ax.scatter(2**pwrOf2, fftw8*1e6, label = 'pyFFTW interface - 8 threads',alpha = 0.5)
ax.scatter(2**pwrOf2, fftw16*1e6, label = 'pyFFTW interface - 16 threads',alpha = 1)
ax.scatter(2**pwrOf2, mkl*1e6, label = 'Intel MKL',alpha = 1)
ax.scatter(2**pwrOf2, mkl_par*1e6, label = 'Intel MKL (array of '+str(L)+'xL)',alpha = 1)
if includeCupy:
    ax.scatter(2**pwrOf2, cupy*1e6, label = 'cuPy',alpha = 1)
    ax.scatter(2**pwrOf2, cupy_par*1e6, label = 'cuPy (array of '+str(L)+'xL)',alpha = 1)
ax.scatter(2**pwrOf2, ez_num*1e6, label = 'ezfft - numpy',alpha = 0.5)
ax.scatter(2**pwrOf2, ez_sci*1e6, label = 'ezfft - scipy',alpha = 0.5)
ax.scatter(2**pwrOf2, ez_fftw*1e6, label = 'ezfft - fftw',alpha = 0.5)
ax.scatter(2**pwrOf2, ez_mkl*1e6, label = 'ezfft - mkl',alpha = 0.5)
ax.scatter(2**pwrOf2, ez_cup*1e6, label = 'ezfft - cupy',alpha = 0.5)
ax.set_yscale('log')
ax.set_xscale('log', basex=2)
plt.legend(loc = 'right',bbox_to_anchor=(0.60, 0.82))
plt.ylabel(r'Average FFT runtime [$\mu$s]')
#plt.ylim(0.1,10**4)
plt.xlabel('Vector length L')