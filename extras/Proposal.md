# Modelado de dinámica de contagio de SARS-CoV-2 en EE. UU. 

## Integrantes del grupo

- Zappala Carolina ()

- Gesualdi Santiago (@SantiiiGesualdi)

## Descripción

El objetivo de este proyecto es modelar el mecanismo de contagio y propagación del virus SARS-CoV-2 sobre la población de los distintos estados de Estados Unidos, teniendo en cuenta el modelo epidemiológico _MSEIR_ y la vacunación en adultos.

## Dataset

El [dataset a analizar](https://github.com/CSSEGISandData/COVID-19/tree/master) sobre casos de coronavirus en Estados Unidos es el dado por Johns Hopkins University [^1].

Se puede llegar a recurrir a otros datasets para el testeo del modelo y para su entrenamiento, sean estos sobre los casos de COVID-19 en EE. UU. o no.

## Consideraciones

- Se espera poder recurrir a la construcción de una _DDE_ (Delay Differential Equation) teniendo en consideración un período de incubación del COVID-19 y que existen modelos epidemiológicos "clásicos" que involucran este factor, así como la inmunidad pasiva y demás clasificaciones de los agentes.

- Según lo que podamos conversar, podría ser interesante no solo ver el funcionamiento a una "escala intraestatal", sino notar una vinculación entre estos mismos. Siendo una opción modelar el sistema como un grafo y recurrir a técnicas de _Graph Neural ODE's_ para modelar los parámetros.

## Método propuesto (preliminar)

La premisa es partir de un modelo básico epidemiológico como lo es _SIR_ (muy similar, cualitativamente, a _Lotka-Volterra_) a partir de datos sintéticos y migrar a modelados cada vez más complejos para utilizar los datos reales recopilados. Se propone utilizar redes neuronales para modelar las variables caóticas del sistema puesto que, dadas las condiciones del análisis y la velocidad de propagación y reacción ante la enfermedad, no se encuentran por solución "válida" a estos modelos resultados estacionarios.

Resultaría de interés ver qué tan viables son las simplificaciones realizadas y, en caso de ser necesario (y poder encuadrarlo en el alcance del proyecto), modelar la dimensión interestatal que vincula las relaciones entre los estados para mejorar las aproximaciones del sistema a partir de grafos, lo que permitiría convivir en un espacio de parámetros rotundamente superior.

Para cualquiera sea el caso, resulta vital la utilización de los conocimientos propios de la dinámica y de las leyes que lo rigen; por eso se espera recurrir a _PINN's_ para poder escalar el modelo a uno de mayor envergadura y con índices superiores de verosimilitud.


[^1]: Dong E, Du H, Gardner L. **An interactive web-based dashboard to track COVID-19 in real time**. The Lancet Infectious Diseases, 2020; 20, 533-534.
https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30120-1/fulltext