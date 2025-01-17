# Universidad de Costa Rica 
## Escuela de Ingeniería Eléctrica
### Modelos Probabilísticos de Señales y Sistemas
#### Tarea 04 
#### Esteban Arias Vásquez-B50677


## Situación
En el archivo bits10k.csv se encuentran 10.000 bits (actualizado) generados a partir de una fuente binaria equiprobable. El objetivo es hacer una modulación digital para "transmitir" estos datos por un canal ruidoso. La modulación se hace con una frecuencia en la portadora de f = 5000 Hz y con un período de símbolo igual a un período completo de la onda portadora.
## Asignaciones 
1) (20 %) Crear un esquema de modulación BPSK para los bits presentados. Esto implica asignar una forma de onda sinusoidal normalizada (amplitud unitaria) para cada bit y luego una concatenación de todas estas formas de onda.
2) (10 %) Calcular la potencia promedio de la señal modulada generada.
3) (20 %) Simular un canal ruidoso del tipo AWGN (ruido aditivo blanco gaussiano) con una relación señal a ruido (SNR) desde -2 hasta 3 dB.
4) (10 %) Graficar la densidad espectral de potencia de la señal con el método de Welch (SciPy), antes y después del canal ruidoso.
5) (20 %) Demodular y decodificar la señal y hacer un conteo de la tasa de error de bits (BER, bit error rate) para cada nivel SNR.
6) (20 %) Graficar BER versus SNR.

## Solución:
1) Lo primero que se debe hacer es leer los datos del archivo 'bits10k.csv' y guardarlos en un array de python. Luego es importante definir todas las constantes como la frecuencia (5 kHz), el período de la onda (el cuál va a ser el tiempo de duración de cada ciclo de la onda portadora, T = 1/f) y el número de muestras por período ( p = 50). Luego de definir esto, se debe tener en claro en que consiste la modulación BSPK, la cuál basicamente consta de dos formas de onda que dependen de la entrada. Para una entrada igual a 1 (bits = 1) la onda portadora toma una forma senoidal con la frecuencia definida anteriormente, en el periodo definido anteriormente y con el número de muestras que se definió también, esto se logra creando un vector tp con linspace que va desde o hasta T, con p cantidad de puntos. Por otro lado para una entrada igual a 0 (bits = 0) la onda portadora toma la forma de una onda seno pero en este caso negativa, por lo que se utiliza la misma onda creada anteriormente con la diferencia de que en este caso es negativa. 
La creación de la onda modulada se realiza mediante un for que recorre cada elemento de bits, además se utiliza 'enumerate(bits)' el cual crea pares ordenados con el valor de cada elemento de bits y su número de posición dentro del array. Dentro del for se uriliza un condicional if/else el cual va a darle forma de onda a la señal dependiendo del valor de bits. Es importante mencionar que en la línea de código 'for k, b in enumerate(bits):' k va a ser la posición del elemento b, que estamos leyendo dentro del array bits. 
Luego de esto se grafica la señal modulada para los primeros 5 bits ( senal[ 0: 5* p]). La figura correspondiente a esta solución se encuentra dentro del mismo repositorio con el nombre de 'modulacionBSPK.png'. A continuación se muestra el resultado obtenido para bits con la secuencia: 1,0,1,0,1.
![modulacionBSPK.png](attachment:modulacionBSPK.png)

2) La potencia promedio está dada por: $$ P(T) = \frac{1}{2T} \int_{-T}^{T} \! E[X^2(T)]dt = A\{E[X^2(t)]\} $$ en dónde $E[X^2(T)]$ es la autocorrelación del proceso aleatorio X(t) y A\{\} es el operador de promedio temporal. Entonces, primero calculamos la potencia instantánea como el cuadrado de la señal, y luego se integra dentro del intervalo de tiempo Tx. Acá es necesario importar desde spicy el módulo integrate. 
El resultado final es: P = 0.49 W. 

3) Se parte de la relación señal-ruido: $$ SNR_{dB} = 10log_{10}(\frac{P_s}{P_n}) $$ se crea un vector para SNR que va desde -2 hasta 3. Con cada valor de SNR se calcula la potencia $P_n$ y luego de esto la desviación estándar $$ \sigma = \sqrt{P_n} $$ Con este sigma vamos a crear ruido con la función de numpy random.nomal y se le suma a señal original, luego se grafica. 
Se realiza el mismo procedimiento para cada SNR y se guardan las figuras con los nombres canalruidosoX.png hasta canalruidoso5.png. A continuación se muestra el resultado para SNR = 2 dB. 
![canalruidoso.png](attachment:canalruidoso.png)

4) Se debe importar el módulo signal de la biblioteca scipy, del cual se va a utilizar la función signal.welch al cuál le entran como parámentros un array con la señal de la cual se quiere la densidad espectral, la frecuencia de muestreo fs (250kHz), y un nperseg = 1024 (el valor default es 256) y devuelve dos arrays con los valores de los ejes de la gráfica de la densidad espectral. Entonces se utiliza ésta función para la señal antes y después del ruido, se grafican y se guardan como 'densidadantes.png' y 'densidaddespes.png' dentro del repositorio de la tarea, cabe recalcar que solo se obtiene la densidad espectral después del ruido para SNR = -2dB, si se necesita alguna otra es sólo de cambiar la señal con ruido que ingresa como parámetro en la función. A continuación se muestra el resultado para antes del ruido y después del ruido con SNR = -2 dB.
Antes del ruido:
![densidadantes.png](attachment:densidadantes.png)
Después del ruido: (SNR = -2 dB)
![densidaddespues.png](attachment:densidaddespues.png)

5) Se crea nuevamente un vector con distintos valores para SNR, en este caso de -7 a -2 pues en el intervalo dado para SNR en el punto 3 no existen errores. Luego se realiza un for para calcular la señal con ruido Rx para cada SNR dado, y dentro de este for se usa otro for para que con cada Rx que depende de SNR calcule un BER. Lo que se busca es obtener un vector del para BER en dónde cada valor dependa de SNR. 
Para encontrar BER, se crea una señal 'Es' que es la pseudo-energia de la onda original y va a ser el nivel de comparación con la señal después del ruido. Se pone un umbral en Es/2 y se compara con la energía de la señal después del ruido. 
El resultado es: "Hay un total de 13.0 errores en 9999 bits para una tasa de error de [0.030203020302030203, 0.019001900190019003, 0.011501150115011502, 0.0067006700670067, 0.0025002500250025004, 0.0013001300130013002].] "

6) Se grafican BER versus SNR con plt.semilogy para obtener una gráfica semilogarítmica. Se guarda como 'BERvsSNR.png' en el repositorio de la tarea. El resultado se muestra a continuación:

![BERvsSNR.png](attachment:BERvsSNR.png)


```python

```
