import numpy as np
from util.util import Interpolation
import open3d as o3d
from scipy.spatial import KDTree
from itertools import chain
import pymeshlab as pm
import json

class SkeletonBranch:
    def __init__( self, centers ):
        self.centers = centers
        self.curve = Interpolation( centers )
        self.joints = []
        self.jointMeshVertexIndices = {}
        self.jointSubmesh = {}
        self.jointSubmeshCurvatures = {}

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
    
    def amountOfJoints( self ):
        return len(self.joints)
    
    def extendSubmesh( self, jointIdx, vertices ):
        self.jointMeshVertexIndices[ jointIdx ] = np.append( self.jointMeshVertexIndices[ jointIdx ], vertices )

    def jointDistance( self, jointIdx ):
        return self.curve.arcLength( tEnd=self.joints[ jointIdx ] )
    
    def hasVertices( self, jointIdx ):
        return len(self.jointMeshVertexIndices[jointIdx]) != 0 
    
    def saveSubmesh( self, jointIdx, submesh, curvatures ):
        self.jointSubmesh[ jointIdx ] = submesh
        self.jointSubmeshCurvatures[ jointIdx ] = curvatures

    def centerAndNormalize( self ):
        bases = self.curve.basesAlong( self.joints )

        for jointIndex, (subMesh, basis) in enumerate(zip(self.jointSubmesh.values(), bases)):
            
            subMesh.translate( -self.getJointPosition(jointIndex) )

            B_inv = np.linalg.inv(basis.T) # existe porque son ortogonales entre si -> son li
            T = np.block( [[ B_inv, np.zeros((3,1)) ], [np.zeros((1,3)), np.array([1]) ] ])
            subMesh.transform( T )

            maxCoord = np.max( np.abs( np.asarray(subMesh.vertices) ) )
            subMesh.scale( 1/maxCoord, center=(0,0,0)) # ahora en rango [0,0,0], [2,2,2]


class SkeletonMesh:
    def __init__( self, meshFile, skeletonFile, correspondanceFile ):

        self.mesh = o3d.io.read_triangle_mesh( meshFile )
        self.mesh.compute_vertex_normals( normalized = True )

        self.curvature = self._calculateCurvature( meshFile )
        self.branches = []

        with open( skeletonFile ) as file:
            for line in file:
                contents = line.split(" ")
                self.branches.append( SkeletonBranch( np.reshape( np.array( contents[1:]).astype(np.float32), 
                                                                ((len(contents) - 1) // 3, 3) ) ))
        
        if len( self.branches ) == 0:
            raise ValueError("The skeleton provided is empty")

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

    @property
    def amountOfJoints( self ):
        return self._amountOfJoints
    
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
        for branch in self.branches:
            for jointIndex, submesh in branch.getSubmeshes():
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
                                    'amount_joints': branch.amountOfJoints(),
                                    'joints': [ 
                                    {
                                        'position': branch.getJointPosition(jointIdx).tolist(),
                                        'distance': str(branch.jointDistance(jointIdx)),
                                        'vertices': np.asarray(submesh.vertices).tolist(),
                                        'triangles': np.asarray(submesh.triangles).tolist(),
                                        'normals': np.asarray(submesh.vertex_normals).tolist(),
                                        'curvature': np.asarray(branch.getJointCurvature(jointIdx)).tolist()
                                    }
                                    for jointIdx, submesh in branch.getSubmeshes() if branch.hasVertices(jointIdx)] } for branch in self.branches 
                            ]
                        }, jsonFile
                    )

                stop = False
            except:
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


if __name__=='__main__':
    skel = SkeletonMesh( 
        'data/humans/mesh/test_15070.off', 
        'data/humans/centerline/faust_15070.txt',
        'data/humans/centerline/faust_15070.polylines.txt' )

    skel.submesh()

    #print( list(skel.getSubmeshes()) )