# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 20:41:46 2021

@author: sebas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

"""
Definición de variables:
X1:Concentración de bacterias acidogénicas [g/L]
X2:Concentración de bacterias metanogénicas [g/L]
S1: Concentración de sustrato orgánico [g/L]
S2: Concentración de ácidos grasos volátiles [mmol/L]
Z: Alcalinidad Total [mmol/L]
C: Concentración total de carbono inorgánico en [mmol/L]
CH4: Concentración de metano [mmol/L]
"""

def reactor (t,x):
    #Condiciones iniciales
    X1=x[0]
    X2=x[1]
    S1=x[2]
    S2=x[3]
    Z=x[4]
    C=x[5]
    
    #Datos
    u1max = 1.2 #1/d
    alpha = 0.5
    Ks1 = 7.1 #1/d
    #D = 0.34 #1/d
    u2max = 0.74 #1/d
    Ks2 = 9.28 #mmol/L
    KI2 = 256 #mmol/L
    Ka = 1.5*(10**(-5))*1000 #mol/L a mmol/L
    pHin = 5.12
    S2in = 93.6 #mmol/L
    S1in = 9.5 #g/L
    k1 = 42.14
    k2 = 116.5 #mmol/g
    k3 = 268 #mmol/g
    Cin = 65 #mmol/L
    kla = 19.8 #1/d
    Kh = 32 #mmol/atm*L
    Pt = 1 #atm
    k4 = 50.6 #mmol/g
    k5 = 343.6 #mmol/g
    k6 = 453 #mmol/g
    
    #Ecuaciones auxiliares
    u1 = u1max*(S1/float(S1+Ks1)) #1/d
    u2 = u2max*(S2/float(S2+Ks2+(S2**2/float(KI2)))) #1/d
    Zin = (Ka/float(Ka+(10**(-pHin))))*S2in #mmol/L
    qm = k6*u2*X2 #mmol/L*d
    phi = C+S2-Z+(Kh*Pt)+(qm/float(kla))
    Pc = (phi-np.sqrt((phi**2)-(4*Kh*Pt)*(C+S2-Z)))/float(2*Kh) #atm
    qc = kla*(C+S2-Z-(Kh*Pc)) #mmol/L*d
    
    #Set de ODEs
    dX1dt = (u1-(alpha*D))*X1
    dX2dt = (u2-(alpha*D))*X2
    dS1dt = D*(S1in-S1)-k1*u1*X1
    dS2dt = D*(S2in-S2)+k2*u1*X1-k3*u2*X2
    dZdt = D*(Zin-Z)
    dCdt = D*(Cin-C)-qc+k4*u1*X1+k5*u2*X2
    
    return np.array([dX1dt, dX2dt, dS1dt, dS2dt, dZdt, dCdt])


#Condiciones inicales
X1o = 0.375 #g/L
X2o = 0.375 #g/L
S1o = 1.8 #g/L
S2o = 2.5 #mmol/L
Zo = 62 #mmol/L
Co = 65 #mmol/L

D = 0.34 #1/d

yo =np.array([X1o,X2o,S1o,S2o,Zo,Co])  #Condiciones iniciales modelo dinámico

tiempomax=5
t_span=np.array([0,tiempomax])
times = np.linspace(0,tiempomax,int(tiempomax*1000))

y=solve_ivp(reactor,t_span,yo,t_eval=times,method="Radau")
t=y.t
X1=y.y[0]
X2=y.y[1]
S1=y.y[2]
S2=y.y[3]
Z=y.y[4]
C=y.y[5]

ch4 = []
k6 = 453 #mmol/g
Ks2 = 9.28 #mmol/L
KI2 = 256 #mmol/L
u2max = 0.74 #1/d
for i in range(len(t)):
    u2 = u2max*(S2[i]/float(S2[i]+Ks2+(S2[i]**2/float(KI2)))) #1/d
    ch4last=k6*u2*X2[i]
    ch4.append(ch4last)
    

#Gráficos Modelo Dinámico sin discretizar

plt.figure()
plt.title("Concentración de bacterias acidogénicas")
plt.plot(t,X1,"b-")
plt.xlabel('time, d')
plt.ylabel('X1, g/L')
plt.grid()
plt.show()

plt.figure()
plt.title("Concentración de bacterias metanogénicas")
plt.plot(t,X2,"b-")
plt.xlabel('time, d')
plt.ylabel('X2, g/L')
plt.grid()
plt.show()

plt.figure()
plt.title("Alcalinidad total")
plt.plot(t,Z,"b-")
plt.xlabel('time, d')
plt.ylabel('Z, mmol/L')
plt.grid()
plt.show()


plt.figure()
plt.title("Concentración de sustrato orgánico")
plt.plot(t,S1,"b-")
plt.xlabel('time, d')
plt.ylabel('S1, g/L')
plt.grid()
plt.show()

plt.figure()
plt.title("Concentración de ácidos grasos volátiles")
plt.plot(t,S2,"b-")
plt.xlabel('time, d')
plt.ylabel('S2, mmol/L')
plt.grid()
plt.show()


plt.figure()
plt.title("Concentración total de carbono inorgánico")
plt.plot(t,C,"b-")
plt.xlabel('time, d')
plt.ylabel('C , mmol/L')
plt.grid()
plt.show()


plt.figure()
plt.title("Concentración de metano")
plt.plot(t,ch4,"b-")
plt.xlabel('time, d')
plt.ylabel('CH4, mmol/L')
plt.grid()
plt.show()
