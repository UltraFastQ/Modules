import numpy as np
from scipy.constants import c as C
from scipy.interpolate import interp1d as interp
from scipy.integrate import simps as simpson
import matplotlib.pyplot as plt
import femtoQ.tools as fq



def get_FWHM(t,I_t):
    
    tmp = np.linspace(t[0],t[-1],1000000)
    
    I_tmp = interp(t,I_t,'quadratic')
    I_tmp = I_tmp(tmp)
    
    I_tmp = I_tmp - np.min(I_tmp)
    I_tmp /= np.max(I_tmp)

    t_1 = tmp[I_tmp>=0.5][0]
    t_2 = tmp[I_tmp>=0.5][-1]

    return t_2-t_1

def unpack_data(filename,wavelengthLimits,makeFigures):
    
    data = np.load(filename)
    
    delays = data['time']*1e-15
    wavelengthsSpectro = data['wavelengths']*1e-9
    trace = data['trace']
    trace/= np.max(trace)
    
    trace = trace[:,((wavelengthsSpectro>wavelengthLimits[0])&(wavelengthsSpectro<wavelengthLimits[1]))]
    wavelengthsSpectro = wavelengthsSpectro[((wavelengthsSpectro>wavelengthLimits[0])&(wavelengthsSpectro<wavelengthLimits[1]))]
    
    data.close()
    
    
    if makeFigures:
        plt.figure()
        plt.pcolormesh(delays*1e15,wavelengthsSpectro*1e9,trace.transpose(),shading = 'auto')
        plt.title('Measured trace')
        plt.ylabel('Wavelengths [nm]')
        plt.xlabel('Delay [fs]')
        plt.colorbar()


    
    return delays, wavelengthsSpectro, trace


def plot_output(pulseRetrieved, initialGuess, pulseFrequencies, traceRetrieved, traceFrequencies,delays, wavelengthsSpectro):

    t, E = freq2time(pulseFrequencies, pulseRetrieved)
    
    t, E_cmp = freq2time(pulseFrequencies, np.abs(pulseRetrieved))

    t0 = t[np.argmax(np.abs(E)**2)]
    E = interp(t-t0,E,'quadratic',bounds_error=False,fill_value=0)
    E = E(t)
    
    print('Measured FWHM duration: ' + str( round(get_FWHM(t, np.abs(E)**2)*1e15,1) ) + ' fs')
    print('Fourier-lim. FWHM duration: ' + str( round(get_FWHM(t, np.abs(E_cmp)**2)*1e15,1) ) + ' fs')
    
    plt.figure()
    plt.plot(t*1e15,np.abs(E)**2 / np.max(np.abs(E)**2),'b',label = 'Retrieved')
    plt.plot(t*1e15,np.abs(E_cmp)**2 / np.max(np.abs(E_cmp)**2),'k',label = 'Transform limit')
    plt.xlabel('Time [fs]')
    plt.ylabel('Power [arb. u.]')
    plt.legend()
    
    
    wavelengths = C/(pulseFrequencies)
    II = np.argsort(wavelengths)
    
    
    plt.figure()
    axL = plt.gca()
    axR = axL.twinx()
    axL.plot(pulseFrequencies/1e12,np.abs(initialGuess)**2,'r',linewidth = 3,label = 'Initial guess')
    axL.plot(pulseFrequencies/1e12,np.abs(pulseRetrieved)**2 / np.max(np.abs(pulseRetrieved[II])**2),'k',linewidth = 3,label = 'Retrieved spectrum')
    axL.set_ylabel('Normalized power density')
    axL.set_xlabel('Frequency [THz]')
    axR.set_ylabel('Spectral phase [rad]')
    
    retrievedPhase = np.unwrap(np.angle(pulseRetrieved))
    retrievedPhase -= np.average(retrievedPhase[II],weights = np.abs(pulseRetrieved[II])**2)
    
    IIplot = np.abs(pulseRetrieved)**2 > (np.abs(pulseRetrieved)**2).max()/20
    
    axR.plot(pulseFrequencies[IIplot]/1e12,retrievedPhase[IIplot],'--k')
    plt.xlim(C/(wavelengthsSpectro[-1]*2.5)/1e12, C/(wavelengthsSpectro[0]*1.8)/1e12)
    
    plt.figure()
    plt.pcolormesh(delays*1e15,(C/traceFrequencies[traceFrequencies>0])*1e9,traceRetrieved[:,traceFrequencies>0].transpose() / (traceRetrieved[:,traceFrequencies>0].transpose()).max(),shading='auto')
    plt.title('Retrieved trace')
    plt.ylabel('Wavelengths [nm]')
    plt.xlabel('Delay [fs]')
    plt.ylim(wavelengthsSpectro[0]*1e9,wavelengthsSpectro[-1]*1e9)
    plt.colorbar()

    return axL


def freq2time(frequency, spectrum):
    # Interpolate over new frequency grid, including negative w components
    v_max = frequency[-1]
    v_min = -v_max
    N = len(spectrum)
    dv = v_max/N
    
    new_v = np.hstack((np.array([0]),np.linspace(dv,v_max,N),np.linspace(v_min,-dv,N)))
    
    interp_marginal = interp(frequency,spectrum,'quadratic',bounds_error=False,fill_value=0)
    spectrum = interp_marginal(new_v)

    E = np.fft.fftshift(np.fft.ifft(spectrum))
    t = np.fft.fftshift(np.fft.fftfreq(2*N+1,dv))

    return t, E

def RANA(delays,w_shg,trace_w,w_fund):
    
    a = 0.09
    b = 0.425
    c = 0.1
    
    # Integrate over delay axis
    marginal_w = simpson(trace_w,delays,axis = 0)
    
    marginal_w = fq.ezsmooth(marginal_w,11,'hanning')
    
    marginal_w[marginal_w<0] = 0
    
    # Interpolate over new frequency grid, including negative w components
    w_max = w_shg[-1]*2
    w_min = -w_max
    N = len(marginal_w)*2
    dw = w_max/N
    
    new_w = np.hstack((np.array([0]),np.linspace(dw,w_max,N),np.linspace(w_min,-dw,N)))
    
    interp_marginal = interp(w_shg,marginal_w,'quadratic',bounds_error=False,fill_value=0)
    marginal_w = interp_marginal(new_w)

    S = np.fft.fftshift(np.fft.ifft(marginal_w))
    t = np.fft.fftshift(np.fft.fftfreq(2*N+1,dw/2/np.pi))

    s_p = np.sqrt(S)
    s_m = -np.sqrt(S)
    
    s = np.zeros_like(s_p)
    for ii,tau in enumerate(t):
        
        if ii==0:
            s[ii] = s_m[ii]
            
        if ii==1:
            ds0_p = abs(s_p[ii] - s[ii-1])
            ds0_m = abs(s_m[ii] - s[ii-1])
            
            ds1_p = 0
            ds1_m = 0
            
            ds2_p = 0
            ds2_m = 0
            
        
        if ii==2:
            ds0_p = abs(s_p[ii] - s[ii-1])
            ds0_m = abs(s_m[ii] - s[ii-1])
            
            ds1_p = abs( ds0_p - (s[ii-1] - s[ii-2]) )
            ds1_m = abs( ds0_m - (s[ii-1] - s[ii-2]) )
        
            ds2_p = 0
            ds2_m = 0
        
        else:
            ds0_p = abs(s_p[ii] - s[ii-1])
            ds0_m = abs(s_m[ii] - s[ii-1])
            
            ds1_p = abs( ds0_p - (s[ii-1] - s[ii-2]) )
            ds1_m = abs( ds0_m - (s[ii-1] - s[ii-2]) )
            
            ds2_p = abs( ds1_p - ( (s[ii-1] - s[ii-2]) - (s[ii-2] - s[ii-3]) ) )
            ds2_m = abs( ds1_m - ( (s[ii-1] - s[ii-2]) - (s[ii-2] - s[ii-3]) ) )
            
        e_p = a*ds0_p**2 + b*ds1_p**2 + c*ds2_p**2
        e_m = a*ds0_m**2 + b*ds1_m**2 + c*ds2_m**2
        
        
        if e_p < e_m:
            s[ii] = s_p[ii]
        else:
            s[ii] = s_m[ii]
    
    spectrum = np.fft.fftshift(np.fft.fft(s*np.hanning(len(s))))
    v = np.fft.fftshift((np.fft.fftfreq(2*N+1,t[1]-t[0])))
    spectrum = np.abs(spectrum)**0.5
    spectrum_interp = interp(v*2*np.pi,spectrum,'quadratic',bounds_error=False,fill_value=0)
    spectrum = spectrum_interp(w_fund)
    spectrum -= np.min(spectrum)
    spectrum /= np.max(spectrum)
    
    
    spectrum = np.complex128(spectrum) #* np.exp(1j*np.random.normal(0,np.pi/4,len(spectrum)))
    return spectrum