import numpy as np
from util.util import Interpolation
import open3d as o3d
from scipy.spatial import KDTree
from itertools import chain, combinations
import pymeshlab as pm
import networkx as nx
import json

class SkeletonBranch:
    def __init__( self, centers ):
        self.centers = centers
        self.curve = Interpolation( centers )
        self.joints = []
        self.jointMeshVertexIndices = {}
        self.jointSubmesh = {}
        self.jointSubmeshCurvatures = {}
        self.jointTransformations = {}
        self.direction = 0

    def sampleJoints( self, pointsPerUnit ):
        if len(self.centers) <= 3:
            self.joints = [ 0, 1 ]
        else:
            self.joints = self.curve.sampleByCurvature( pointsPerUnit=pointsPerUnit )

        self.jointMeshVertexIndices = { i: [] for i in range(len(self.joints)) }

        return len(self.joints)

    def submesh( self, skeleton ):
        rangeOfJoint = lambda i : np.ceil( len(self.centers) * (self.joints[i] + (self.joints[i + 1] - self.joints[i])/2) ).astype(np.uint32)
        l = 0
        for i in range(len(self.joints)):
            h = len(self.centers) if i == len(self.joints) - 1 else rangeOfJoint(i)
            
            try:
                self.jointMeshVertexIndices[i] = np.concatenate( [ skeleton.getVerticesOfCenter( center ) for center in self.centers[l:h]] )
            except ValueError:
                pass
                    
            l = h

            yield i, self.jointMeshVertexIndices[ i ]

    def getSubmeshes( self ):
        for jointIdx, submesh in self.jointSubmesh.items():
            yield jointIdx, submesh

    def getJointSubmesh( self, jointIdx ):
        return self.jointSubmesh[ jointIdx ]

    def getJointVertices( self, jointIdx ):
        return self.jointMeshVertexIndices[ jointIdx ]

    def getJoints( self ):
        return (self.curve[joint] for joint in self.joints)
    
    def getJointPosition( self, jointIdx ):
        return self.curve[self.joints[jointIdx]]
    
    def getJointCurvature( self, jointIdx ):
        return self.jointSubmeshCurvatures[ jointIdx ]
    
    def getJointTransformation( self, jointIdx ):
        return self.jointTransformations[jointIdx]
    
    def amountOfJoints( self ):
        return len(self.joints)
    
    def extendSubmesh( self, jointIdx, vertices ):
        self.jointMeshVertexIndices[ jointIdx ] = np.append( self.jointMeshVertexIndices[ jointIdx ], vertices )

    def jointDistance( self, jointIdx ):
        if self.direction == -1:
            return self.curve.arcLength( tStart = 1, tEnd=self.joints[ jointIdx ] )
        else:
            return self.curve.arcLength( tEnd=self.joints[ jointIdx ] )
    
    def jointDistanceToNN( self, jointIdx ):
        if self.direction == 0:
            if jointIdx == 0:
                return self.jointDistance(1)
            elif jointIdx == len(self.joints) - 1:
                return np.abs( self.jointDistance( jointIdx ) - self.jointDistance(jointIdx - 1) )
            else:
                return np.abs( min( self.jointDistance( jointIdx - 1), self.jointDistance( jointIdx + 1)) - self.jointDistance(jointIdx) )
        else:
            if jointIdx == 0:
                return self.jointDistance(0) - self.jointDistance(1)
            elif jointIdx == len(self.joints) - 1:
                return self.jointDistance(jointIdx - 1)
            else:
                return np.abs( min( self.jointDistance( jointIdx - 1), self.jointDistance( jointIdx + 1)) - self.jointDistance(jointIdx) )
    
    def hasVertices( self, jointIdx ):
        return len(self.jointMeshVertexIndices[jointIdx]) != 0 
    
    def saveSubmesh( self, jointIdx, submesh, curvatures ):
        self.jointSubmesh[ jointIdx ] = submesh
        self.jointSubmeshCurvatures[ jointIdx ] = curvatures

    def centerAndNormalize( self ):
        '''
            Supongo la malla esta en la base dada por la curva (como un frenet frame).
            Entonces Bv = w,  donde w son los vertices posta.
            Para obtenerla normalizada entonces hago 
                B_inv B v = B_inv w
                v = B_inv w = w'
            Osea ahora la matriz Id seria nuestra base, que es lo que quiero.
            
            Ademas, para entrenar quiero que esten en el cubo [-1,-1,-1] y [1,1,1].
            Pero quiero que mantengan (por ahora dentro de cada rama exclusivamente) las relaciones de tamaños.
            Entonces tengo que escalar por el más grande.
        '''
        bases = self.curve.basesAlong( self.joints )
        maxCoord = 0
        for (jointIndex, subMesh), basis in zip(self.jointSubmesh.items(), bases):
            
            #subMesh.translate( -self.getJointPosition(jointIndex) )
            T = np.block( [[np.eye(3,3), -1 * self.getJointPosition(jointIndex).reshape(3,1)], [np.zeros(3), np.ones(1)]])

            B_inv = np.linalg.inv(basis.T) # existe porque son ortogonales entre si -> son li
            B_inv = np.block( [[ B_inv, np.zeros((3,1)) ], [np.zeros((1,3)), np.array([1]) ] ])

            H =  B_inv @ T
            subMesh.transform( H )
            self.jointTransformations[jointIndex] = H

            maxCoord = max( maxCoord, np.max( np.abs( np.asarray(subMesh.vertices) ) ))
        
        for jointIndex, subMesh in self.jointSubmesh.items():
            S = np.block( [ [np.eye(3,3) * (1/maxCoord), np.zeros((3,1))], [np.zeros((1,3)), np.ones(1)]])
            subMesh.transform( S )

            self.jointTransformations[jointIndex] = S @ self.jointTransformations[jointIndex]
    
    def direct( self, parent ):
        if (np.linalg.norm(parent.getJointPosition( parent.amountOfJoints() - 1 ) - self.getJointPosition( len(self.joints) - 1 )) <
            np.linalg.norm(parent.getJointPosition( parent.amountOfJoints() - 1 ) - self.getJointPosition( 0 )) ):
            self.direction = -1
            return len(self.joints) - 1
        
        return 0

class SkeletonMesh:
    def __init__( self, meshFile, skeletonFile, correspondanceFile ):

        self.mesh = o3d.io.read_triangle_mesh( meshFile )
        self.mesh.compute_vertex_normals( normalized = True )

        self.curvature = self._calculateCurvature( meshFile )
        self.branches = []
        self.skipableJoints = {}

        borders = {}
        centers = []
        with open( skeletonFile ) as file:
            for idx, line in enumerate(file):
                line =line.replace('\n', '')
                contents = line.split(" ")

                if tuple(contents[1:4]) in borders:
                    borders[ tuple(contents[1:4]) ].add(idx)
                else:
                    borders[ tuple(contents[1:4]) ] = set([idx])

                if tuple(contents[-3:]) in borders:
                    borders[ tuple(contents[-3:]) ].add(idx)
                else:
                    borders[ tuple(contents[-3:]) ] = set([idx])

                centers.append( np.reshape( np.array( contents[1:]).astype(np.float32), ((len(contents) - 1) // 3, 3) ) )

        
        treeOfCenters = KDTree( np.concatenate(centers))
        for idx, branchCenters in enumerate(centers):
            query = treeOfCenters.query_ball_point( branchCenters, r=0.01, return_length=True)

            if not all( quantity > 1 for quantity in query ):
                self.branches.append( SkeletonBranch( branchCenters ))
            else:
                print("SALTO")

        

        if len( self.branches ) == 0:
            raise ValueError("The skeleton provided is empty")
        
        self.skeletonGraph = nx.Graph()
        self.skeletonGraph.add_nodes_from( list(range(0,len(self.branches))) )
        self.skeletonGraph.add_edges_from( [ t for s in borders.values() for t in combinations(s, 2) ] )

        self.skeletonRoot = int(np.argmin( np.max( [ [ len(k) for k in d.values() ] for d in dict(nx.all_pairs_shortest_path(self.skeletonGraph)).values() ], axis=1 ) ))
        self.skeletonTree = nx.bfs_tree( self.skeletonGraph, self.skeletonRoot )

        self.treeOfVertices = KDTree( self.vertices )
        self._amountOfJoints = 0
        self._verticesOfCenter = {}
        with open(correspondanceFile) as file:
            for line in file:
                s = line.split(' ')
                center = np.array(s[1:4]).astype(np.float32)

                vertex = np.array(s[4:]).astype(np.float32)
                vertexIndex = self.treeOfVertices.query( vertex, k = 1 )[1]
                
                if tuple(center) in self._verticesOfCenter:
                    self._verticesOfCenter[tuple(center)].append(vertexIndex)
                else:
                    self._verticesOfCenter[tuple(center)] = [vertexIndex]

    @property
    def vertices( self ):
        return np.asarray( self.mesh.vertices )

    @property
    def triangles( self ):
        return np.asarray( self.mesh.triangles )

    def amountOfBranches( self ):
        return len(self.branches)

    @property
    def amountOfJoints( self ):
        return self._amountOfJoints
    
    def getBranches( self ):
        for branchNumber in self.skeletonTree.nodes:
            yield branchNumber, self.branches[branchNumber]

    def branchParents( self, branchIdx ):
        return nx.shortest_path( self.skeletonTree, source=self.skeletonRoot, target=branchIdx )[:-1]

    def getVerticesOfCenter( self, center ):
        if tuple(center) not in self._verticesOfCenter:
            self._verticesOfCenter[tuple(center)] = []
        
        return self._verticesOfCenter[tuple(center)]

    def submesh( self, pointsPerUnit=5 ):
        self._amountOfJoints = 0
        for branch in self.branches:            
            self._amountOfJoints += branch.sampleJoints( pointsPerUnit )

            for jointIdx, submeshIndices in branch.submesh( self ):
                self.generateSubmesh(branch, jointIdx, submeshIndices)

    def generateSubmesh(self, branch, jointIdx, submeshIndices):
        newMesh = o3d.geometry.TriangleMesh( self.mesh  )
        vertexMask = np.array([i not in submeshIndices for i in np.arange(len(np.asarray(self.mesh.vertices)))])
        newMesh.remove_vertices_by_mask( vertexMask )
        branch.saveSubmesh( jointIdx, newMesh, self.curvature[ ~vertexMask ] )

    def getJoints( self ):
        return chain.from_iterable( branch.getJoints() for branch in self.branches )

    def getSubmeshes( self ):
        for branchIdx, branch in enumerate(self.branches):
            for jointIndex, submesh in branch.getSubmeshes():
                if branchIdx in self.skipableJoints and self.skipableJoints[branchIdx] != jointIndex:
                    yield branch.getJointPosition(jointIndex), submesh, branch.getJointCurvature( jointIndex )

    def branchAndIndexOfJoint( self, t ):
        for branch in self.branches:
            amountOfJoints = len( branch.joints )
            if t - amountOfJoints < 0:
                return branch, t
            else:
                t -= amountOfJoints

    def postprocess( self, alpha=0.5 ):
        treeOfJoints = KDTree( list(self.getJoints()) )

        for index, joint in enumerate(self.getJoints()):
            branch, jointIdx = self.branchAndIndexOfJoint( index )

            radiusOfMIS = self.treeOfVertices.query( joint, k=1 )[0]
            neighborJoints = treeOfJoints.query_ball_point( joint, radiusOfMIS * alpha )

            change = False
            for neighborJoint in neighborJoints:
                if neighborJoint != index:
                    change = True
                    branchOfNeighbor, neighborJointIndex = self.branchAndIndexOfJoint( neighborJoint )
                    branch.extendSubmesh( jointIdx, branchOfNeighbor.getJointVertices( neighborJointIndex ) )

            if change:
                submeshIndices = branch.getJointVertices( jointIdx )
                self.generateSubmesh( branch, jointIdx, submeshIndices)

        for branchIdx in self.skeletonTree.nodes:
            parents = list(self.skeletonTree.in_edges( branchIdx ))
            if len(parents) == 1:
                jointConnection = self.branches[branchIdx].direct( self.branches[ parents[0][0]] )
                self.skipableJoints[branchIdx] = jointConnection

    def saveToJson( self, path ):
        filePath= path
        fileName = path[:path.rfind('.')]
        i = 1
        stop = True
        while stop:
            try:
                with open(filePath, 'x') as jsonFile:
                    json.dump(
                        {
                            'amount_branches': len(self.branches),
                            'branches' : [
                                {
                                    'branch_number': branchNumber,
                                    'parents': self.branchParents(branchNumber),
                                    'amount_joints': branch.amountOfJoints(),
                                    'joints': [ 
                                    {
                                        'position': branch.getJointPosition(jointIdx).tolist(),
                                        'distance': str(branch.jointDistance(jointIdx) / branch.jointDistance(branch.amountOfJoints() - 1) ),
                                        'vertices': np.asarray(submesh.vertices).tolist(),
                                        'triangles': np.asarray(submesh.triangles).tolist(),
                                        'normals': np.asarray(submesh.vertex_normals).tolist(),
                                        'curvature': np.asarray(branch.getJointCurvature(jointIdx)).tolist(),
                                        'transformation': np.linalg.inv(branch.getJointTransformation(jointIdx)).tolist(),
                                        'mean': calculateMean( self.amountOfBranches(), self.branchParents(branchNumber), branchNumber, branch.jointDistance(jointIdx) / branch.jointDistance(branch.amountOfJoints() - 1) ),
                                        'cov': calculateCovMatrix( self.amountOfBranches(), branchNumber, branch.jointDistanceToNN( jointIdx ) / branch.jointDistance(branch.amountOfJoints() - 1) / 6)
                                    }
                                    for jointIdx, submesh in branch.getSubmeshes() if branch.hasVertices(jointIdx) ] } for branchNumber, branch in self.getBranches()
                            ]
                        }, jsonFile, default=str
                    )

                stop = False
            except FileExistsError:
                filePath = f'{fileName}({i}).json'
                i += 1

        return filePath

    def centerAndNormalize( self ):
        for branch in self.branches:
            branch.centerAndNormalize()

    def _calculateCurvature( self, meshFile ):
        pyMeshset = pm.MeshSet()
        pyMeshset.load_new_mesh( meshFile )
        pyMesh = pyMeshset.current_mesh()

        d = pyMeshset.apply_filter("compute_scalar_by_discrete_curvature_per_vertex", curvaturetype='Mean Curvature')
        pyMeshset.compute_new_custom_scalar_attribute_per_vertex(name="v_curv", expr="q")
        v_curv = pyMesh.vertex_custom_scalar_attribute_array('v_curv')
        return np.clip( v_curv, a_min = -1 * float(d['90_percentile']), a_max = float(d['90_percentile']))

def calculateMean( size, parents, branchNumber, distance):
    mean = np.zeros(size)
    np.put( mean, np.concatenate( [ parents,  [branchNumber]] ).astype(np.int64), np.concatenate( [np.ones_like(parents), [distance]] ) )
    return mean.tolist()

def calculateCovMatrix( size, branchNumber, std ):    
    mat = np.zeros( (size,size) ) + np.eye(size,size) * 1e-10 # pytorch me pide que sea positiva definida... le pongo valor muy chico
    mat[branchNumber, branchNumber] = std
    return mat.tolist()

if __name__=='__main__':
    
    skel = SkeletonMesh( 
        'data/humans/mesh/test_15070.off', 
        'data/humans/centerline/faust_15070.txt',
        'data/humans/centerline/faust_15070.polylines.txt' )
    skel.submesh()

    #print( list(skel.getSubmeshes() ))