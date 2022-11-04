import networkx as nx
import numpy as np
from scipy.optimize import minimize
from scipy import integrate
from util import Vec3, Interpolada

class GrafoCentros:
    '''
        Clase grafo de centerline
    '''
    def __init__( self, grafo ):
        self.G = nx.convert_node_labels_to_integers( grafo )
        if nx.number_connected_components( grafo ) != 1:
            raise ValueError( "El grafo tiene mas de 1 componente conexa" )

        self.maxNombre = self.cantNodos( ) - 1

    def direccion( self, nodoFrom, nodoTo ):
        return self.posicionNodo( nodoFrom ).dirTo( self.posicionNodo(nodoTo) )
    
    def vecinos( self, nodo ):
        return ( list(arista)[1] for arista in self.G.edges(nodo) )
    
    def vecinosDistintos( self, nodo, nodosDist ):
        return [ vecino for vecino in self.vecinos(nodo) if not vecino in nodosDist ]

    def iesimoVecino( self, nodo, i ):
        return list(self.vecinos( nodo ))[i]

    def nodoMasCercano( self, nodo, listaNodos ):
        return listaNodos[ np.argmin([ self.posicionNodo(nodo).distTo( self.posicionNodo(otro) ) for otro in listaNodos ] ) ] 

    def vecinoMasCercano( self, nodo ):
        return self.nodoMasCercano( nodo, list( self.vecinos(nodo) ))
    
    def cantNodos( self ):
        return len(self.nodos())

    def nodos( self ):
        return self.G.nodes

    def posicionNodo( self, nodo ):
        # cuando arme el nodo pongo la posicion como un Vec3
        posicion = nx.get_node_attributes( self.G, 'posicion' )[nodo]
        if not isinstance(posicion, Vec3):
            nx.set_node_attributes( self.G, {nodo: Vec3(*posicion)}, 'posicion')
            posicion = nx.get_node_attributes( self.G, 'posicion' )[nodo]

        return posicion

    def radioNodo( self, nodo ):
        return nx.get_node_attributes( self.G, 'radio' )[nodo]

    def gradoNodo( self, nodo ):
        return self.G.degree( nodo )

    def getNuevoNombreNodo( self ):
        self.maxNombre += 1
        return self.maxNombre

    def elegirNodoGrado( self, grado ):
        for nodo in self.nodos():
            if self.gradoNodo(nodo) == grado:
                return nodo

    def crearNodo( self, posicion, radio ):
        nombre = self.getNuevoNombreNodo()
        self.G.add_node( nombre, posicion=posicion, radio=radio)
        return nombre
    
    def crearArista( self, nodoOrigen, nodoFin ):
        self.G.add_edge( nodoOrigen, nodoFin )
    
    def eliminarNodo( self, nodo ):
        self.G.remove_node( nodo )

    def obtenerRamasDesdeNodo( self, nodoInicial, nodoProcedencia=None ):
        '''
            Devuelvo los nodos de una rama, partiendo de un nodo inicial, que presunpongo de grado 1 o n > 2.
        '''
        ramas = []
        nodoPrevio = nodoInicial
        for nodoActual in self.vecinos(nodoPrevio):
            if not nodoProcedencia is None and nodoActual == nodoProcedencia:
                continue
            
            nodosRama = [ nodoPrevio ]

            while self.gradoNodo(nodoActual) == 2:
                nodosRama.append(nodoActual)
                nodoProximo = self.vecinosDistintos( nodoActual, [ nodoPrevio ] )[0]
                nodoPrevio = nodoActual
                nodoActual = nodoProximo

            nodosRama.append(nodoActual)

            ramas.append(nodosRama)
            nodoPrevio = nodoInicial

        return ramas

    def grafoDeRamas( self ):
        grafo = nx.Graph()

        for nodo in self.nodos():
            if self.gradoNodo( nodo ) == 1 or self.gradoNodo( nodo ) > 2:
                grafo.add_node( nodo )

        for nodo in grafo.nodes:
            ramas = self.obtenerRamasDesdeNodo( nodo )
            for rama in ramas:
                if not (nodo, rama[-1]) in grafo.edges:
                    grafo.add_edge(nodo, rama[-1], rama=rama )

        return grafo

    def resamplear( self, *, alpha=0.1, beta=0.1, w=0.01, puntosPorUnidad=0.3 ):
        grafoRamas = self.grafoDeRamas( )
        diccionarioRamas = nx.get_edge_attributes(grafoRamas, 'rama')

        for edge in grafoRamas.edges:
            self.resamplearRama( diccionarioRamas[edge], alpha, beta, w, puntosPorUnidad )

        self.G = nx.convert_node_labels_to_integers( self.G )

    def resamplearRama( self, listaNodos, alpha, beta, w, puntosPorUnidad ):
        
        posicionesNodos = [ self.posicionNodo( nodo ) for nodo in listaNodos ]
        curvaPosicionesInterpolada = Interpolada(  posicionesNodos ).reparametrizar( lambda x : np.clip(x, 0, 1))

        radioNodos = [ self.radioNodo( nodo ) for nodo in listaNodos ]
        radiosInterpolados = Interpolada( radioNodos ).reparametrizar( lambda x : np.clip(x, 0, 1))
        
        curvaturaInterpolada = curvaPosicionesInterpolada.curvatura().reparametrizar( lambda x : np.clip(x, 0, 1))

        termino = lambda j : radiosInterpolados[j] / (1 + beta * curvaturaInterpolada[j])
        
        def costo( ts ):
            
            sigmoide = lambda x : 1 / (1 + np.exp(-10*x))
            f = lambda x, y: np.exp( -x + y )
            h = lambda x, y: -x + f(0,y)
            g = lambda x, y: h(x,y) * sigmoide(-x) + f(x,y) * sigmoide(x)

            def CostoPrincipal( xs ):
                return np.sum( 
                    [ g( xs[0], 1 / len(xs) )]+
                    [ g( xs[i+1] - xs[i], 1 / len(xs) ) for i in range(0, len(xs)-1)] +
                    [ g( 1 - xs[-1], 1 / len(xs) )] )

            def CostoSecundario( xs ):
                return np.sum( 
                    [ g( xs[0], alpha * (termino(0) + termino(xs[0])) )]+
                    [ g( xs[i+1] - xs[i], alpha* (termino(xs[i]) + termino(xs[i+1])) ) for i in range(0, len(xs)-1)] +
                    [ g( 1 - xs[-1], alpha*(termino(xs[-1]) + termino(1) ) )] )

            
            return CostoPrincipal( ts ) + w * CostoSecundario( ts )
        
        cantPuntos = self.estimadorCantPuntos( curvaPosicionesInterpolada, puntosPorUnidad )
        paso = 1 / cantPuntos
        ts = np.linspace(0 + paso, 1 - paso, cantPuntos)
        parametros = minimize( costo, ts )        

        self.actualizarRama( listaNodos, curvaPosicionesInterpolada.evaluarLista(parametros.x), radiosInterpolados.evaluarLista(parametros.x) )

    @staticmethod
    def curvaInterpoladaConBordes( puntos, bordeIzq, bordeDer, cantPuntos ):
        radioNodos = np.concatenate( [ bordeIzq, puntos, bordeDer  ] )
        primerIndice = ( 1 / len(radioNodos) ) * cantPuntos
        ultimoIndice = ( 1 / len(radioNodos) ) * ( cantPuntos + len(puntos) )
        return Interpolada( radioNodos ).reparametrizar( lambda x : (ultimoIndice - primerIndice) * x + primerIndice ), ( -primerIndice / (ultimoIndice - primerIndice) + 0.01, (1-primerIndice) / (ultimoIndice - primerIndice) - 0.01)
    
    @staticmethod
    def estimadorCantPuntosViejo( h, alpha, grado=5 ):
        integral = integrate.quad(h, 0, 1)
        ak = integrate.newton_cotes( grado )[0]
        return np.max( [1, int( ((1 + alpha * (h(0) - h(1))) * np.max(ak)) / (2 * alpha * integral[0]) )])

    @staticmethod
    def estimadorCantPuntos( curva, puntosPorUnidad ):
        return np.max( [1, int(curva.longitudDeArco() * puntosPorUnidad ) ] )

    def actualizarRama( self, nodosARemplazar, nuevasPosiciones, nuevosRadios ):
        [ self.eliminarNodo( nodo ) for nodo in nodosARemplazar[1:-1] ] # elimino los nodos menos los de las puntas

        ultimoNodo = nodosARemplazar[0]
        for posicion, radio in zip( nuevasPosiciones, nuevosRadios):
            nodoNuevo = self.crearNodo( posicion, radio )
            self.crearArista( ultimoNodo, nodoNuevo )
            ultimoNodo = nodoNuevo

        self.crearArista( ultimoNodo, nodosARemplazar[-1] )

        return ultimoNodo

