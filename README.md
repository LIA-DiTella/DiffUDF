# Pipeline tesis

<details>
    <summary>Extracción esqueleto</summary>

Malla original             |  Esqueleto extraido
:-------------------------:|:-------------------------:
<img src="resources/mesh.png" height="400">|<img src="resources/skeleton.png" height="400">

    Logramos realizar la extracción utilizando el método implementado en CGAL.
  
</details>

<details>
    <summary>Sampleo de articulaciones esqueleto</summary>

Esqueleto original         |  Articulaciones sampleadas
:-------------------------:|:-------------------------:
<img src="resources/skeleton.png" height="400">|<img src="resources/sampleo.png" height="400">

    Pienso samplear utilizando el método que ya implementé que tiene en cuenta la curvatura y el largo de cada curva.
  
</details>

<details>
    <summary>Matching de vertices a articulaciones</summary>

Articulaciones sampleadas  |  Matching
:-------------------------:|:-------------------------:
<img src="resources/sampleo.png" height="400">|<img src="resources/matching.png" height="400">

    En primera instancia pienso simplemente utilizar los vértices más cercanos dentro de algún rango.
    Esto sin dudas que trae problemas (ejemplo hombro). Habrá que ver si son muy graves
  
</details>

<details>
    <summary>Entrenamiento</summary>

    Ya lo pude hacer para una unica curva. Faltaría ver como hacer cuando tenemos multiples curvas.
    Posibles ideas son:
        - Agregar más parametros intrínsecos.
        - Agregar selectores de curvas.
  
</details>

<details>
    <summary>Rendering</summary>

    Posiblemente el desafio más grande.
    Esta bueno que como primer intento alcanza con usar marching cubes como en la demo chiquita que hice.
    Es un problema que no esta resuelto a nivel de una única malla y nuestro desarrollo solo complica las cosas. Así que creo que quedará para otro trabajo optimizar y lograr renders pro que utilicen ray marching o cosas por el estilo.
  
</details>
