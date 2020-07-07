# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 19:54:27 2020

@author: Esteban
"""

""" Tarea 04 Modelos Probabilísticos de Señales y Sistemas
Esteban Arias Vásquez 
B50677 """
 
#Importamos bibliotecas 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from scipy import integrate 
from scipy import signal

"""
1) Crear un esquema de modulación BPSK para los bits presentados. Esto implica asignar una forma de onda sinusoidal 
normalizada (amplitud unitaria) para cada bit y luego una concatenación de todas estas formas de onda.

"""
#Importamos los datos del archivo 'bits10k.csv' en un vector'
datos = pd.read_csv('bits10k.csv')
bits = np.array(datos)


#Inicializamos las constantes
N = len(bits)
f = 5000 #Hz, frecuencia según encabezado de tarea
T = 1/f #Período de cada símbolo (onda)
p = 50 # Número de puntos de muestreo por período
tp = np.linspace(0, T, p) # Vector de 0 a T, con p puntos (vector de muestreo del período)

"""Ahora creamos la forma de onda para la modulación.
Modulación BSPK: si el bit es uno la señal de la onda portadora
va a ser seno, 
si es un 0 se desfasa 180 grados, o sea para bit = 0 la
señal portadora es -seno.
"""
sinus = np.sin(2*np.pi*f*tp)
#Visualización de la forma de onda 
"""
plt.plot(tp, sinus)
plt.plot(tp, sinus1)
plt.title('-sen')
plt.show()
"""
#Frecuencia de muestreos
fs = p/T # 250 kHz
#Creamos la linea temporal para toda la señal Tx
t = np.linspace(0, N*T, N*p)
# Se inicializa el vector de la señal modulada, en un vector de ceros
senal = np.zeros(t.shape) #mismo tamaño que t, pues se piensan graficar luego

#Creación de la señal modulada BSPK
#Si bits = 1 --->senal = sinus
# Si bits = 0 ---> senal = -sinus
for k, b in enumerate(bits):
    if b == 1:
        senal[k*p:(k+1)*p] = sinus
    else:
        senal[k*p:(k+1)*p] = -sinus
 #Graficamos los la senal correspondiente a los primeros 5 bits       
plt.figure()
plt.plot(senal[0:5*p])
plt.ylabel('Amplitud señal portadora')
plt.xlabel('Tiempo (ms)')
plt.title('Modulación BSPK para los primeros 5 bits del archivo "bits10k.csv"')
plt.savefig('modulacionBSPK.png')
plt.show


""" 2) Calcular la potencia promedio de la señal modulada generada.
"""
#La potencia instantánea está dada por
Pinst = senal**2

# Potencia promedio a partir de la potencia instantanea
Ps = integrate.trapz(Pinst, t)/(N*T) #W
print('La potencia promedio es: ', Ps , 'W')
#Resultado: P = 0.49 W




""" 3) Simular un canal  ruidoso del tipo AWGN (ruido aditivo blanco gaussiano) con una relación 
señal a ruido (SNR) desde -2 hasta 3 dB.
 """
 #Se realiza el mismo procedimiento para cada SNR
# Potencia del ruido para SNR y potencia de la señal dadas
SNR0 = -2
SNR1 = -1
SNR2 = 0 
SNR3 = 1
SNR4 = 2
SNR5 = 3


#######SNR = -2
Pn0 = Ps / (10**(SNR0/ 10))
# Desviación estándar del ruido
sigma0 = np.sqrt(Pn0)
    
# Crear ruido (Pn = sigma^2)
ruido0 = np.random.normal(0, sigma0, senal.shape)
    
# Simular "el canal": señal recibida
Rx0 = senal + ruido0
     
# Visualización de los  primeros bits recibidos con ruido
pb = 5
plt.figure()
plt.plot(Rx0[0:pb*p])
plt.title('Simulación de un canal ruidoso del tipo AWGN, con SNR -2 dB')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud')
plt.savefig('canalruidoso.png')
plt.show()


#######SNR = -1
Pn1 = Ps / (10**(SNR1/ 10))
# Desviación estándar del ruido
sigma1 = np.sqrt(Pn1)
    
# Crear ruido (Pn = sigma^2)
ruido1 = np.random.normal(0, sigma1, senal.shape)
    
# Simular "el canal": señal recibida
Rx1 = senal + ruido1
     
# Visualización de los  primeros bits recibidos con ruido
pb = 5
plt.figure()
plt.plot(Rx0[0:pb*p])
plt.title('Simulación de un canal ruidoso del tipo AWGN, con SNR -1 dB')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud')
plt.savefig('canalruidoso1.png')
plt.show()


#######SNR = 0
Pn2 = Ps / (10**(SNR2/ 10))
# Desviación estándar del ruido
sigma2 = np.sqrt(Pn2)
    
# Crear ruido (Pn = sigma^2)
ruido2 = np.random.normal(0, sigma2, senal.shape)
    
# Simular "el canal": señal recibida
Rx2 = senal + ruido2
     
# Visualización de los  primeros bits recibidos con ruido
pb = 5
plt.figure()
plt.plot(Rx2[0:pb*p])
plt.title('Simulación de un canal ruidoso del tipo AWGN, con SNR 0 dB')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud')
plt.savefig('canalruidoso2.png')
plt.show()


#######SNR = 1
Pn3 = Ps / (10**(SNR3/ 10))
# Desviación estándar del ruido
sigma3 = np.sqrt(Pn3)
    
# Crear ruido (Pn = sigma^2)
ruido3 = np.random.normal(0, sigma3, senal.shape)
    
# Simular "el canal": señal recibida
Rx3 = senal + ruido3
     
# Visualización de los  primeros bits recibidos con ruido
pb = 5
plt.figure()
plt.plot(Rx3[0:pb*p])
plt.title('Simulación de un canal ruidoso del tipo AWGN, con SNR 1 dB')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud')
plt.savefig('canalruidoso3.png')
plt.show()


#######SNR = 2
Pn4 = Ps / (10**(SNR4/ 10))
# Desviación estándar del ruido
sigma4 = np.sqrt(Pn4)
    
# Crear ruido (Pn = sigma^2)
ruido4 = np.random.normal(0, sigma4, senal.shape)
    
# Simular "el canal": señal recibida
Rx4 = senal + ruido4
     
# Visualización de los  primeros bits recibidos con ruido
pb = 5
plt.figure()
plt.plot(Rx4[0:pb*p])
plt.title('Simulación de un canal ruidoso del tipo AWGN, con SNR 2 dB')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud')
plt.savefig('canalruidoso4.png')
plt.show()


#######SNR = 3
Pn5 = Ps / (10**(SNR5/ 10))
# Desviación estándar del ruido
sigma5 = np.sqrt(Pn5)
    
# Crear ruido (Pn = sigma^2)
ruido5 = np.random.normal(0, sigma5, senal.shape)
    
# Simular "el canal": señal recibida
Rx5 = senal + ruido5
     
# Visualización de los  primeros bits recibidos con ruido
pb = 5
plt.figure()
plt.plot(Rx5[0:pb*p])
plt.title('Simulación de un canal ruidoso del tipo AWGN, con SNR 3 dB')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud')
plt.savefig('canalruidoso5.png')
plt.show()

"""4) Graficar la densidad espectral de potencia de la señal con el método de Welch (SciPy),.
 antes y después del canal ruidoso.
"""
# Antes del canal ruidoso
fw, PSD = signal.welch(senal, fs, nperseg=1024)
plt.figure()
plt.semilogy(fw, PSD)
plt.xlabel('Frecuencia / Hz')
plt.ylabel('Densidad espectral de potencia / V**2/Hz')
plt.savefig('densidadantes.png')
plt.show()
#Se grafica solo para SNR = -2 dB
# Después del canal ruidoso
fw, PSD = signal.welch(Rx0, fs, nperseg=1024)
plt.figure()
plt.semilogy(fw, PSD)
plt.xlabel('Frecuencia / Hz')
plt.ylabel('Densidad espectral de potencia / V**2/Hz')
plt.savefig('densidaddespues.png')
plt.show()

"""5) Demodular y decodificar la señal y hacer un conteo de la tasa de error de bits 
(BER, bit error rate) para cada nivel SNR.""" 
# Pseudo-energía de la onda original (esta es suma, no integral)
Es = np.sum(sinus**2)
#inicializamos un nuevo SNR 
#Se inicia en valores distintos a los dados en el enunciado para lograr obtener errores 
#pues en el intervalo del enunciado se obtienen 0 errores

SNR1 = [-7,-6,-5,-4,-3,-2]
# Inicialización del vector de bits recibidos
bitsRx = np.zeros(bits.shape)
#vector para BER del mismo tamaño de SNR 
BER = [0,0,0,0,0,0]
#for para calcular los Rx que dependen de SNR
for n in range(len(SNR1)):
    Pn = Ps / (10**(SNR1[n]/ 10))
    # Desviación estándar del ruido
    sigma = np.sqrt(Pn)
    
    # Crear ruido (Pn = sigma^2)
    ruido = np.random.normal(0, sigma, senal.shape)
    
    # Simular "el canal": señal recibida
    Rx = senal + ruido

# Decodificación de la señal por detección de energía para cada Rx que depende de cada SNR
    for k, b in enumerate(bits):
        Ep = np.sum(Rx[k*p:(k+1)*p] * sinus) 
        if Ep > Es/2:
            bitsRx[k] = 1
        else:
            bitsRx[k] = 0

    err = np.sum(np.abs(bits - bitsRx))
    BER[n] = err/N
    

print('Hay un total de {} errores en {} bits para una tasa de error de {}.'.format(err, N, BER))

"""6) Graficar BER  versus SNR.
"""

plt.figure() 
plt.semilogy(SNR1, BER[0:5*p])
plt.title('BER versus SNR')
plt.ylabel('BER')
plt.xlabel('SNR en dB')
plt.savefig('BERvsSNR.png')
plt.show()
 
 