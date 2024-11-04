#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 19:43:01 2024

@author: Gabriele Lovicu
"""

get_ipython().magic('reset -sf')

from math import exp, sqrt, cos, sin
import numba
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from NVpy import eigenState, nitrogenVacancy
matplotlib.rcParams['text.usetex'] = True  #install these dependencies: sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super

# Transition rates kij [Hz]
k41 = 62.7e6;
k52 = 62.7e6;
k63 = 62.7e6;
k57 = 80e6;
k67 = 80e6;
k47 = 12.97e6;
k71 = 3.45e6;
k72 = 1.08e6;
k73 = 1.08e6;

#Constants
h = 6.62607015e-34;
hbar = h/(2*np.pi);
c = 299792458;        
D1 = 2.87e9;
D2 = 1.42e9;
en_4 = h*c/637e-9;
mu0 = 4*np.pi*1e-7;
elementaryCharge = 1.60217663e-19; #[C]
Nsim = 300 #number of simulations

rateMatrix = np.zeros([7,7])
rateMatrix[3,0] = k41
rateMatrix[4,1] = k52
rateMatrix[5,2] = k63
rateMatrix[4,6] = k57
rateMatrix[5,6] = k67
rateMatrix[3,6] = k47
rateMatrix[6,0] = k71
rateMatrix[6,1] = k72
rateMatrix[6,2] = k73

energy = np.zeros(6);
energy[0] = 0+1e-40;
energy[1] = h*D1;
energy[2] = h*D1;
energy[3] = en_4;
energy[4] = en_4 + h*D2;
energy[5] = en_4 + h*D2;

W1_ref = np.zeros(Nsim)
W2_ref = np.zeros(Nsim)
W3_ref = np.zeros(Nsim)
W4_ref = np.zeros(Nsim)
W5_ref = np.zeros(Nsim)
W6_ref = np.zeros(Nsim)

W1_sig = np.zeros(Nsim)
W2_sig = np.zeros(Nsim)
W3_sig = np.zeros(Nsim)
W4_sig = np.zeros(Nsim)
W5_sig = np.zeros(Nsim)
W6_sig = np.zeros(Nsim)

Nref = np.zeros(Nsim)
Nsig = np.zeros(Nsim)
C = np.zeros(Nsim)
Nc = np.zeros(Nsim)

duration_RFpulse_array = np.zeros(Nsim)

wavelength4 = np.zeros(Nsim)
wavelength5 = np.zeros(Nsim)
wavelength6 = np.zeros(Nsim)

frequency1 = np.zeros(Nsim)
frequency2 = np.zeros(Nsim)
frequency3 = np.zeros(Nsim)

frequency4 = np.zeros(Nsim)
frequency5 = np.zeros(Nsim)
frequency6 = np.zeros(Nsim)

for j in range(Nsim):
    
    NV = nitrogenVacancy(energy)
    
    OmegaR_0 = 1.5e7
    OmegaR_p = OmegaR_0
    Wp_0 = 1.9e6;
    Gamma2 = 5e5
    
    initialState = np.array([1/3, 1/3, 1/3, 0, 0, 0, 0, 0])
    
    duration_laserPulse = 10e-6
    duration_depletion = 1.5e-6;
    duration_RFpulse = 0+j*10e-9
    duration_RFpulse_array[j] = 0+j*4e-9
    duration = duration_laserPulse + duration_depletion + duration_RFpulse
    Nt = int(1e4)
    
    numberCycles = 2
    Wp = Wp_0*np.zeros(Nt)
    OmegaR = np.zeros(Nt)#1.5e7 #[rad/s]
    Wp_array = []
    OmegaR_array = []
    
    Wp[0:int(Nt*duration_laserPulse/duration)] = [Wp_0]*int(Nt*duration_laserPulse/duration);
    OmegaR[int(Nt*(duration_laserPulse+duration_depletion)/duration):Nt]= [OmegaR_p]*(Nt-int(Nt*(duration_laserPulse+duration_depletion)/duration))
    
    for i in range(numberCycles):
        Wp_array = np.append(Wp_array, Wp)
        OmegaR_array = np.append(OmegaR_array, OmegaR)
    
    delta_t = duration/Nt
    t = np.linspace(0,numberCycles*duration,numberCycles*Nt) 
    
    currentState = NV.evolution(Wp_array, OmegaR_array, Gamma2, initialState, rateMatrix, numberCycles*duration, numberCycles*Nt)

    Nc[j] = np.sum(NV.coherencePopulation[int(Nt*((numberCycles-1)*duration+0*duration_laserPulse)/duration):int(Nt*((numberCycles-1)*duration+0.1*duration_laserPulse)/duration)])

    W4_sig[j] = np.sum(NV.state4.population[int(Nt*((numberCycles-1)*duration+0.70*duration_laserPulse)/duration):int(Nt*((numberCycles-1)*duration+1*duration_laserPulse)/duration)])
    W5_sig[j] = np.sum(NV.state5.population[int(Nt*((numberCycles-1)*duration+0.70*duration_laserPulse)/duration):int(Nt*((numberCycles-1)*duration+1*duration_laserPulse)/duration)])
    W6_sig[j] = np.sum(NV.state6.population[int(Nt*((numberCycles-1)*duration+0.70*duration_laserPulse)/duration):int(Nt*((numberCycles-1)*duration+1*duration_laserPulse)/duration)])
    
    W4_ref[j] = np.sum(NV.state4.population[int(Nt*((numberCycles-1)*duration+0.00*duration_laserPulse)/duration):int(Nt*((numberCycles-1)*duration+0.30*duration_laserPulse)/duration)])
    W5_ref[j] = np.sum(NV.state5.population[int(Nt*((numberCycles-1)*duration+0.00*duration_laserPulse)/duration):int(Nt*((numberCycles-1)*duration+0.30*duration_laserPulse)/duration)])
    W6_ref[j] = np.sum(NV.state6.population[int(Nt*((numberCycles-1)*duration+0.00*duration_laserPulse)/duration):int(Nt*((numberCycles-1)*duration+0.30*duration_laserPulse)/duration)])

    
    Nsig[j] = W4_sig[j]+W5_sig[j]+W6_sig[j];
    Nref[j] = W4_ref[j]+W5_ref[j]+W6_ref[j];
    C[j] = np.abs(Nref[j]-Nsig[j])/Nref[j]#np.max(Nref[j]+Nsig[j])
    
    print(j/Nsim)

plt.figure()
plt.plot(duration_RFpulse_array, C)
plt.xlabel('RF pulse duration [$\mu$s]')
plt.ylabel('Contrast')
plt.rcParams.update({'font.size': 16})
