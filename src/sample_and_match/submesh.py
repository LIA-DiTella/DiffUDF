import numpy as np
from interpol import Interpolation
import open3d as o3d
from scipy.spatial import KDTree
from itertools import chain
import networkx as nx

class SkeletonBranch:
    def __init__( self, centers ):
        self.centers = centers
        self.curve = Interpolation( centers )
        self.joints = []
        self.jointMeshVertices = {}

    def sampleJoints( self, pointsPerUnit ):
        if len(self.centers) <= 3:
            self.joints = [ 0, 1 ]
        else:
            self.joints = self.curve.sampleByCurvature( pointsPerUnit=pointsPerUnit )

        self.jointMeshVertices = { joint: [] for joint in self.joints }

        return len(self.joints)

    def submesh( self, skeleton ):
        rangeOfJoint = lambda i : np.ceil( len(self.centers) * (self.joints[i] + (self.joints[i + 1] - self.joints[i])/2) ).astype(np.uint32)
        l = 0
        for i in range(len(self.joints)):
            h = len(self.centers) if i == len(self.joints) - 1 else rangeOfJoint(i)
            
            try:
                self.jointMeshVertices[self.joints[i]] = np.concatenate( [ skeleton.getVerticesOfCenter( center ) for center in self.centers[l:h]] )
            except ValueError:
                print(f'The joint {i} of branch {self} has no vertices.')
                    
            l = h

    def getSubmeshes( self ):
        for joint, vertIndices in self.jointMeshVertices.items():
            yield (joint, vertIndices)

    def getSubmeshOfJoint( self, jointIdx ):
        return self.jointMeshVertices[self.joints[jointIdx]]

    def getJoints( self ):
        return (self.curve[joint] for joint in self.joints)
    
    def getNJoint( self, n ):
        return self.curve[self.joints[n]]
    
    def extendSubmesh( self, jointIdx, vertices ):
        self.jointMeshVertices[self.joints[jointIdx]] = np.append( self.jointMeshVertices[self.joints[jointIdx]], vertices )

class SkeletonMesh:
    def __init__( self, meshFile, skeletonFile, correspondanceFile, epsilon=0.01 ):

        self.mesh = o3d.io.read_triangle_mesh( meshFile )
        self.branches = []

        with open( skeletonFile ) as file:
            for line in file:
                contents = line.split(" ")
                self.branches.append( SkeletonBranch( np.reshape( np.array( contents[1:]).astype(np.float32), 
                                                                ((len(contents) - 1) // 3, 3) ) ))
                
        self.treeOfVertices = KDTree( self.vertices )
        self._verticesOfCenter = {}
        self._amountOfJoints = 0
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
        return self._verticesOfCenter[tuple(center)]

    def sampleJoints( self, pointsPerUnit=5 ):
        self._amountOfJoints = 0
        for branch in self.branches:            
            self._amountOfJoints += branch.sampleJoints( pointsPerUnit )
            branch.submesh( self )

    def getJoints( self ):
        return chain.from_iterable( branch.getJoints() for branch in self.branches )

    def getSubmeshIndices( self ):
        return chain.from_iterable( branch.getSubmeshes() for branch in self.branches )

    def getSubmeshes( self ):
        for joint, submeshIndices in self.getSubmeshIndices():
            newMesh = o3d.geometry.TriangleMesh( self.mesh  )
            newMesh.remove_vertices_by_mask( [i not in submeshIndices for i in np.arange(len(np.asarray(self.mesh.vertices)))] )
            yield joint, newMesh

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
            branch, jointIndex = self.branchAndIndexOfJoint( index )

            radiusOfMIS = self.treeOfVertices.query( joint, k=1 )[0]
            neighborJoints = treeOfJoints.query_ball_point( joint, radiusOfMIS * alpha )

            for neighborJoint in neighborJoints:
                if neighborJoint != index:
                    branchOfNeighbor, neighborJointIndex = self.branchAndIndexOfJoint( neighborJoint )
                    branch.extendSubmesh( jointIndex, branchOfNeighbor.getSubmeshOfJoint( neighborJointIndex ) )

if __name__=='__main__':
    skel = SkeletonMesh( 
        'data/humans/mesh/test_15070.off', 
        'data/humans/centerline/faust_15070.txt',
        'data/humans/centerline/faust_15070.polylines.txt' )

    skel.sampleJoints()

    #print( list(skel.getSubmeshes()) )