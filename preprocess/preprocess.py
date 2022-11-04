import networkx as nx
import numpy as np
from scipy.optimize import minimize
from util import Interpolation

def straightPaths( graph, startingNode, skip=None ):
    '''
        Returns paths P1, P2, ... Pn where Pi = v1 v2 ... vm,
        with v1 := startingNode, deg(vi) = 2 for i = 2 ... m-1, deg(vm) = 1 or deg(vm) > 3

        skip array-like of nodes => if w in skip then the path Pj = v1 v2 ... vm with v2 == w is skipped
    '''
    paths = []
    lastNode = startingNode
    for currentNode in nx.neighbors( graph, lastNode ):
        if skip is not None and currentNode in skip:
            continue
        
        currentPath = [ lastNode ]

        while graph.degree(currentNode) == 2:
            currentPath.append(currentNode)
            neighbors = list(nx.neighbors( graph, currentNode ))
            nextNode = neighbors[0] if neighbors[0] != lastNode else neighbors[1]
            lastNode = currentNode
            currentNode = nextNode

        currentPath.append(currentNode)

        paths.append(currentPath)
        lastNode = startingNode

    return paths

def collapsedGraph( graph ):
    '''
        Returns simplified graph with collapse of node of degree 2
    '''
    newGraph = nx.Graph()

    for node in graph.nodes:
        if graph.degree( node ) == 1 or graph.degree( node ) > 2:
            newGraph.add_node( node )

    for node in newGraph.nodes:
        paths = straightPaths( graph, node )
        for path in paths:
            if not (node, path[-1]) in newGraph.edges:
                newGraph.add_edge(node, path[-1], path=path )

    return newGraph

def resample( graph : nx.Graph, *, alpha=0.1, beta=0.1, w=0.01, pointsPerUnit=0.3 ):
    if nx.number_connected_components( graph ) != 1:
            raise ValueError( "The graph has more than one connected component" )

    graph = nx.convert_node_labels_to_integers( graph )

    cGraph = collapsedGraph( graph )
    pathDict = nx.get_edge_attributes(cGraph, 'path')

    for edge in cGraph.edges:
        resamplePath( graph, pathDict[edge], alpha, beta, w, pointsPerUnit )

    return graph

def pointAmountEstimator( curve, pointsPerUnit ):
    return np.max( [1, int(curve.arcLength() * pointsPerUnit ) ] )

def resamplePath( graph : nx.Graph, path, alpha, beta, w, pointsPerUnit ):
    nodeCoords = [ nx.get_node_attributes( graph, 'position' )[ node ] for node in path]

    curvePositions = Interpolation( nodeCoords ).reparametrize( lambda x : np.clip(x, 0, 1))
    curvature = curvePositions.curvature().reparametrize( lambda x : np.clip(x, 0, 1))

    term = lambda j : 1 / (1 + beta * curvature[j])
    
    def costo( ts ):
        
        sigmoid = lambda x : 1 / (1 + np.exp(-10*x))
        f = lambda x, y: np.exp( -x + y )
        h = lambda x, y: -x + f(0,y)
        g = lambda x, y: h(x,y) * sigmoid(-x) + f(x,y) * sigmoid(x)

        def principalCost( xs ):
            '''
                forces order of parameters and equidistance
            '''
            return np.sum( 
                [ g( xs[0], 1 / len(xs) )]+
                [ g( xs[i+1] - xs[i], 1 / len(xs) ) for i in range(0, len(xs)-1)] +
                [ g( 1 - xs[-1], 1 / len(xs) )] )

        def secondaryCost( xs ):
            '''
                forces grouping near high curvature areas
            '''
            return np.sum( 
                [ g( xs[0], alpha * (term(0) + term(xs[0])) )]+
                [ g( xs[i+1] - xs[i], alpha* (term(xs[i]) + term(xs[i+1])) ) for i in range(0, len(xs)-1)] +
                [ g( 1 - xs[-1], alpha*(term(xs[-1]) + term(1) ) )] )

        
        return principalCost( ts ) + w * secondaryCost( ts )
    
    cantPuntos = pointAmountEstimator( curvePositions, pointsPerUnit )
    step = 1 / cantPuntos
    ts = np.linspace(0 + step, 1 - step, cantPuntos)
    parameters = minimize( costo, ts )        

    updatePath( graph, path, curvePositions.evaluateFrom(parameters.x) )

def updatePath( graph, nodesToReplace, newCoords):
    graph.remove_nodes_from( nodesToReplace[1:-1] )

    lastNode = nodesToReplace[0]
    for position in newCoords :
        newNode = addNode( graph, position )
        graph.add_edge( lastNode, newNode )
        lastNode = newNode

    graph.add_edge( lastNode, nodesToReplace[-1] )

    return lastNode

def addNode( graph, position ):
    label = len(graph.nodes)
    graph.add_node( label, position=position )
    return label