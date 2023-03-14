import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import pymeshlab as pm
import networkx as nx
import json

def createGraph( branches: list, vertexTree: KDTree ) -> nx.Graph:
    graph = nx.Graph()
    for branch in branches:
        lastNode = None
        for joint in branch:
            
            nodePosition = np.array(joint).astype(np.float32)
            nodeCode = tuple(nodePosition)
            graph.add_node( nodeCode, position=nodePosition, mis_radius=vertexTree.query(nodePosition, k=1)[0] )
            if lastNode != None:
                graph.add_edge( nodeCode, lastNode )

            lastNode = nodeCode

    return graph

def calculateCurvature( meshFile ):
    pyMeshset = pm.MeshSet()
    pyMeshset.load_new_mesh( meshFile )
    pyMesh = pyMeshset.current_mesh()

    d = pyMeshset.apply_filter("compute_scalar_by_discrete_curvature_per_vertex", curvaturetype='Mean Curvature')
    pyMeshset.compute_new_custom_scalar_attribute_per_vertex(name="v_curv", expr="q")
    v_curv = pyMesh.vertex_custom_scalar_attribute_array('v_curv')
    return np.clip( v_curv, a_min = -1 * float(d['90_percentile']), a_max = float(d['90_percentile']))

def computeJoints( graph, alpha, beta ):

    root = None
    for node in graph.nodes:
        if (graph.degree(node) == 1 and 
            all([ graph.degree(neighbors) <= 2 for neighbors in nx.dfs_tree(graph, node, depth_limit= max( beta, np.floor( ((2**alpha) * graph.nodes[node]['mis_radius']))))] )):
            root = node
            break

    if root is None:
        raise ValueError('Couldnt find suitable root')

    tree = nx.dfs_tree( graph, source=root )
    sel = [0]
    parents = [None]

    graphOfJoints = nx.DiGraph()
    for node in tree.nodes:
        if sel[0] == 0:
            graphOfJoints.add_node( node, position=graph.nodes[node]['position'] )
            if parents[0] is not None:
                graphOfJoints.add_edge( parents[0], node )

            sel[0] = max( beta, np.floor( ((2**alpha) * graph.nodes[node]['mis_radius'])))
            parents[0] = node

        if tree.out_degree( node ) > 1:
            sel = [sel[0] - 1] * tree.out_degree( node ) + sel[1:]
            parents = [parents[0]] * (tree.out_degree( node ) - 1) + parents
        elif tree.out_degree(node) == 0:
            graphOfJoints.add_node( node, position=graph.nodes[node]['position'] )
            if parents[0] is not None:
                graphOfJoints.add_edge( parents[0], node )
            sel.pop(0)
            parents.pop(0)
        else:
            sel[0] -= 1

    return graphOfJoints, root

def calculateDistributions( graph, root, std ):
    lastIndex = 0
    indexes = { }
    for node in nx.dfs_tree( graph, root ):
        indexes[node] = lastIndex
        if graph.out_degree( node ) != 1:
            lastIndex += 1

    maxPerIndex = np.zeros( lastIndex )
    for node in nx.bfs_tree( graph, root ):
        if node == root:
            nx.set_node_attributes( graph, {node: np.zeros( lastIndex )}, 'mean')
        else:
            parent = list(graph.in_edges(node))[0][0]
            parentMean = graph.nodes[parent]['mean']
            distanceToParent = np.linalg.norm( graph.nodes[parent]['position'] - graph.nodes[node]['position'])
            nodeMean = parentMean + np.eye(1, len(parentMean), k=indexes[node]).squeeze() * distanceToParent
            nx.set_node_attributes( graph, {node: nodeMean}, 'mean')        
            maxPerIndex[indexes[node]] = max( nodeMean[indexes[node]], maxPerIndex[indexes[node]])

    for node in graph.nodes:
        if any( np.isclose(maxPerIndex, 0) ):
            print('wtf')
        graph.nodes[node]['mean'] /= maxPerIndex

    for node in graph.nodes:
        maxDistance = np.max( np.concatenate([
            [ np.linalg.norm( graph.nodes[u]['mean'] - graph.nodes[v]['mean'], ord=1 ) for u,v in graph.out_edges(node) ],
            [ np.linalg.norm( graph.nodes[u]['mean'] - graph.nodes[v]['mean'], ord=1 ) for u,v in graph.in_edges(node) ]
        ]))
        cov = np.eye( lastIndex, lastIndex) * 1e-20
        cov[indexes[node],indexes[node]] = maxDistance / std

        nx.set_node_attributes( graph, {node: cov}, 'cov')

def divideVertices( graph, joints, verticesOfCenter, alpha, beta ):
    vertexIndicesOfJoint = { joint : set() for joint in joints }
    for joint in joints:
        for center in nx.dfs_preorder_nodes( graph, joint, max( beta, np.floor( ((2**alpha) * graph.nodes[joint]['mis_radius']))) / 2 ):
            if center in verticesOfCenter:
                vertexIndicesOfJoint[joint].update( verticesOfCenter[center] )

    return vertexIndicesOfJoint


def genSubmeshes( mesh, indices, joint, curvature ):
    newMesh = o3d.geometry.TriangleMesh( mesh  )
    vertexMask = np.array([i not in indices[joint] for i in np.arange(len(np.asarray(mesh.vertices)))])
    newMesh.remove_vertices_by_mask( vertexMask )

    return newMesh, curvature[ ~vertexMask ]


def getBases( graphOfJoints, root, upVector ):
    bases= { }
    upVectors = [upVector]
    tree = nx.dfs_tree( graphOfJoints, root)
    for joint in tree.nodes:
        base = computeBase( graphOfJoints, joint, upVectors[0] / np.linalg.norm(upVectors[0]))

        bases[joint] = np.array(base).T

        if tree.out_degree( joint ) > 1:
            upVectors = [base[0]] * (tree.out_degree( joint ) - 1) + upVectors
        elif tree.out_degree(joint) == 0:
            upVectors.pop(0)
        else:
            upVectors[0] = base[0]

    return bases

def computeBase( tree, node, lastUpVector ):
    nodePosition = tree.nodes[node]['position']
    parent = list(tree.in_edges(node))[0][0] if len(list(tree.in_edges(node))) != 0 else None
    if parent is None:
        nodeDirection = np.mean( [ tree.nodes[child]['position'] - nodePosition for _, child in list(tree.out_edges( node )) ], axis=0 )
    else:
        incomingDirection = nodePosition - tree.nodes[parent]['position']
        childs = list(tree.out_edges( node ))
        if len(childs) == 0:
            nodeDirection = incomingDirection
        else:
            nodeDirection = incomingDirection + np.mean( [ tree.nodes[child]['position'] - nodePosition for _, child in  childs], axis=0 )

    nodeDirection = nodeDirection / np.linalg.norm(nodeDirection)
    # proyecto al plano cuya normal es nodeDirection
    if np.isclose( lastUpVector @ nodeDirection, 0):
        upVector = lastUpVector
    else:
        upVector = lastUpVector - nodeDirection * ( np.dot(lastUpVector, nodeDirection) / np.dot(nodeDirection, nodeDirection   ) )
        upVector = upVector / np.linalg.norm(upVector)

    conormal = np.cross( upVector , nodeDirection )
    return upVector, nodeDirection , conormal
    

def normalizeMeshes( graphOfJoints, submeshes, bases ):
    max_coord = 0
    transformations = {}
    for joint, (submesh, curv) in submeshes.items():
        B_inv = np.block( [ [np.linalg.inv( bases[joint] ), np.zeros((3,1))], [np.eye(1,4,k=3)]])
        T = np.block( [ [np.eye(3,3), -1 * graphOfJoints.nodes[joint]['position'].reshape((3,1))], [np.eye(1,4,k=3)]])

        submesh.transform( B_inv @ T )
        transformations[joint] = B_inv @ T

        max_coord = max( np.max( np.abs( np.asarray(submesh.vertices))) , max_coord )

    for joint, (submesh, curv) in submeshes.items():
        S = np.block([ [np.eye(3,3) * (1 / max_coord), np.zeros((3,1))], [np.eye(1,4,k=3)] ]) 
        submesh.transform( S )
        transformations[joint] = S @ transformations[joint]

    return transformations    

def saveToJson( path, graph, submeshes, transformations ):
    filePath= path
    fileName = path[:path.rfind('.')]
    i = 1
    stop = True
    while stop:
        try:
            with open(filePath, 'x') as jsonFile:
                json.dump(
                    {
                        'amount_joints': len(graph.nodes),
                        'joints': [ 
                        {
                            'vertices': np.asarray(submeshes[joint][0].vertices).tolist(),
                            'triangles': np.asarray(submeshes[joint][0].triangles).tolist(),
                            'normals': np.asarray(submeshes[joint][0].vertex_normals).tolist(),
                            'curvature': submeshes[joint][1].tolist(),
                            'mean': graph.nodes[joint]['mean'].tolist(),
                            'cov': graph.nodes[joint]['cov'].tolist(),
                            'transformation': np.linalg.inv(transformations.get( joint, np.eye(4,4))).tolist()
                        }
                        for joint in graph.nodes ]
                        
                    }, jsonFile, default=str
                )

            stop = False
        except FileExistsError:
            filePath = f'{fileName}({i}).json'
            i += 1

    return filePath

def createJson( path, meshFile, skeletonFile, correspondanceFile, alpha=7.5, beta=15, normalize=True, std=6 ):

    mesh = o3d.io.read_triangle_mesh( meshFile )
    mesh.compute_vertex_normals( normalized = True )

    vertexTree = KDTree( np.asarray(mesh.vertices) )
    curvature = calculateCurvature( meshFile )
    centers = []

    with open( skeletonFile ) as file:
        for idx, line in enumerate(file):
            line =line.replace('\n', '')
            contents = line.split(" ")
            centers.append( [ contents[i:i+3] for i in range(1,len(contents),3)] )

    graph = createGraph( centers, vertexTree )
    graphOfJoints, root = computeJoints( graph, alpha=alpha, beta=beta )

    calculateDistributions( graphOfJoints, root, std )

    verticesOfCenter = {}
    with open(correspondanceFile) as file:
        for line in file:
            s = line.split(' ')
            center = np.array(s[1:4]).astype(np.float32)

            vertex = np.array(s[4:]).astype(np.float32)
            vertexIndex = vertexTree.query( vertex, k = 1 )[1]
            
            if tuple(center) in verticesOfCenter:
                verticesOfCenter[tuple(center)].append(vertexIndex)
            else:
                verticesOfCenter[tuple(center)] = [vertexIndex]

    vertexIndicesOfJoint = divideVertices( graph, graphOfJoints.nodes, verticesOfCenter, alpha, beta )

    submeshes = { joint: genSubmeshes(mesh, vertexIndicesOfJoint, joint, curvature) for joint in graphOfJoints.nodes }
    

    bases = getBases(  graphOfJoints, root, np.eye(1,3,k=1).squeeze() )

    return graphOfJoints, bases

    transformations = {}
    if normalize:
        transformations = normalizeMeshes(graphOfJoints, submeshes, bases)

    return saveToJson( path, graphOfJoints, submeshes, transformations)
