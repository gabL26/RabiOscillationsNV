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
matplotlib.rcParams['text.usetex'] = True  #install these dependencies: sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super


class eigenState(object):
    def __init__(self, population, wavelength):
        self.population = population #[adim, normalized]
        self.wavelength = wavelength #[m]

class nitrogenVacancy(object):
    def __init__(self):
        self.state1 = eigenState(0,0)
        self.state2 = eigenState(0,0)
        self.state3 = eigenState(0,0)
        self.state4 = eigenState(0,0)
        self.state5 = eigenState(0,0)
        self.state6 = eigenState(0,0)
        self.state7 = eigenState(0,0)
        self.coherencePopulation = 0;
    
    def evolution(self, Wp, OmegaR, Gamma2, initialState, rateMatrix, duration, Nt):
        self.state1.population = np.zeros(Nt)
        self.state2.population = np.zeros(Nt)
        self.state3.population = np.zeros(Nt)
        self.state4.population = np.zeros(Nt)
        self.state5.population = np.zeros(Nt)
        self.state6.population = np.zeros(Nt)
        self.state7.population = np.zeros(Nt)
        self.coherencePopulation = np.zeros(Nt)
        
        self.state1.population[0] = initialState[0]
        self.state2.population[0] = initialState[1]
        self.state3.population[0] = initialState[2]
        self.state4.population[0] = initialState[3]
        self.state5.population[0] = initialState[4]
        self.state6.population[0] = initialState[5]
        self.state7.population[0] = initialState[6]
        self.coherencePopulation[0] = initialState[7]
        
        delta_t = duration/Nt #[s] discretization step
        
        k41 = rateMatrix[3,0]
        k52 = rateMatrix[4,1]
        k63 = rateMatrix[5,2]
        k57 = rateMatrix[4,6]
        k67 = rateMatrix[5,6]
        k47 = rateMatrix[3,6]
        k71 = rateMatrix[6,0]
        k72 = rateMatrix[6,1]
        k73 = rateMatrix[6,2]
        
        # Update equations
        for i in range(Nt-1):
            self.state1.population[i+1] = self.state1.population[i] + delta_t*(-self.state1.population[i]*Wp[i]+
                                                                               self.state4.population[i]*k41 +
                                                                               self.state7.population[i]*k71 +
                                                                               self.coherencePopulation[i]*OmegaR[i])
                                                                               
            self.state2.population[i+1] = self.state2.population[i] + delta_t*(-self.state2.population[i]*Wp[i]+
                                                                               self.state5.population[i]*k52 +
                                                                               self.state7.population[i]*k72-
                                                                               self.coherencePopulation[i]*OmegaR[i])
            
            self.state3.population[i+1] = self.state3.population[i] + delta_t*(-self.state3.population[i]*Wp[i]+
                                                                               self.state6.population[i]*k63+
                                                                               self.state7.population[i]*k73)
            
            self.state4.population[i+1] = self.state4.population[i] + delta_t*(self.state1.population[i]*Wp[i]-
                                                                               self.state4.population[i]*k41-
                                                                               self.state4.population[i]*k47)
            
            self.state5.population[i+1] = self.state5.population[i] + delta_t*(self.state2.population[i]*Wp[i]-
                                                                               self.state5.population[i]*k52 -
                                                                               self.state5.population[i]*k57)
            
            self.state6.population[i+1] = self.state6.population[i] + delta_t*(self.state3.population[i]*Wp[i]-
                                                                               self.state6.population[i]*k63 -
                                                                               self.state6.population[i]*k67)
            
            self.state7.population[i+1] = 1 - self.state1.population[i+1] - self.state2.population[i+1] - self.state3.population[i+1] - self.state4.population[i+1] - self.state5.population[i+1] - self.state6.population[i+1]
            
            self.coherencePopulation[i+1] = self.coherencePopulation[i] + delta_t*(-Gamma2*self.coherencePopulation[i]+(OmegaR[i]/2)*(self.state2.population[i]-self.state1.population[i]))
        
        return np.array([self.state1.population[Nt-1], self.state2.population[Nt-1], self.state3.population[Nt-1], self.state4.population[Nt-1], self.state5.population[Nt-1], self.state6.population[Nt-1], self.state7.population[Nt-1], self.coherencePopulation[Nt-1]])

#Transition rates
k41 = 62.7e6 #[Hz]
k52 = 62.7e6 #[Hz]
k63 = 62.7e6 #[Hz]
k57 = 80e6 #[Hz]
k67 = 80e6 #[Hz]
k47 = 12.97e6 #[Hz]
k71 = 3.45e6 #[Hz]
k72 = 1.08e6 #[Hz]
k73 = 1.08e6 #[Hz]

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

NV = nitrogenVacancy()

OmegaR_0 = 1.57e7 #[Hz]
Wp_0 = 1.9e6; #[Hz]
Gamma2 = 5e5 #[1/s]

initialState = np.array([1/3, 1/3, 1/3, 0, 0, 0, 0, 0])

duration_laserPulse = 10e-6; #[s] laser pulse
duration_depletion = 1e-6; #[s] depletion time
duration_RFpulse = 1e-6 #[s] Rabi pulse
duration = duration_laserPulse + duration_depletion + duration_RFpulse
Nt = int(1e5)

numberCycles = 2
Wp = Wp_0*np.zeros(Nt)
OmegaR = OmegaR_0*np.zeros(Nt)#1.5e7 #[rad/s]
Wp_array = []
OmegaR_array = []

Wp[0:int(Nt*duration_laserPulse/duration)] = [Wp_0]*int(Nt*duration_laserPulse/duration);
OmegaR[int(Nt*(duration_laserPulse+duration_depletion)/duration):Nt]= [OmegaR_0]*(Nt-int(Nt*(duration_laserPulse+duration_depletion)/duration))

for i in range(numberCycles):
    Wp_array = np.append(Wp_array, Wp)
    OmegaR_array = np.append(OmegaR_array, OmegaR)

delta_t = duration/Nt
t = np.linspace(0,numberCycles*duration,numberCycles*Nt) 


currentState = NV.evolution(Wp_array, OmegaR_array, Gamma2, initialState, rateMatrix, numberCycles*duration, numberCycles*Nt)

plt.figure()#figsize=(10, 6))
plt.plot(t*1e6, NV.state1.population)
plt.plot(t*1e6, NV.state2.population)
plt.plot(t*1e6, NV.state3.population)
plt.plot(t*1e6, NV.state4.population)
plt.plot(t*1e6, NV.state5.population)
plt.plot(t*1e6, NV.state6.population)
plt.plot(t*1e6, NV.state7.population)
plt.plot(t*1e6, NV.coherencePopulation)
plt.rcParams.update({'font.size': 16})
plt.legend(['$n_1$', '$n_2$', '$n_3$', '$n_4$', '$n_5$', '$n_6$', '$n_7$', '$n_c$'])
plt.rcParams.update({'font.size': 16})
plt.xlabel('Time [$\mu$s]')
plt.xlim(right=17)