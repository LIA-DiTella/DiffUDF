import numpy as np
import trimesh as tm
from src.util import normalize
import igl

def calculateCurvature( meshFile ):
    v,_,n, f,_,_ = igl.read_obj(meshFile)
    pd1, pd2, pk1, pk2 = igl.principal_curvature(v, f)

    return v,n,f, (pk1 + pk2)/2

def normalizeFullMesh( mesh ):
    T = np.block( [ [np.eye(3,3), -1 * mesh.center_mass.reshape((3,1))], [np.eye(1,4,k=3)]])

    mesh.apply_transform( T )

    max_coord = np.max( np.abs( mesh.vertices ))

    S = np.block([ [np.eye(3,3) * (1 / (max_coord + max_coord * 0.1)), np.zeros((3,1))], [np.eye(1,4,k=3)] ]) 
    mesh.apply_transform( S )

class ColorVis:
    def __init__(self, colors):
        self.vertex_colors = np.uint8( colors * 255 )
        self.kind = 'vertex'

def preprocessMesh( outputPath, meshFile, not_normalize=True ):
    print('Computing curvatures')
    vertices, normals, triangles, curvatures = calculateCurvature( meshFile )

    # elimino outliers
    curvatures = np.clip( curvatures, np.percentile(curvatures, 5), np.percentile(curvatures, 95) )

    # normalizo curvaturas
    curvatures -= np.min(curvatures)
    curvatures /= np.max(curvatures)

    mesh = tm.Trimesh( vertices, triangles, vertex_normals=normalize(normals) )

    if not not_normalize:
        print('Normalizing')
        normalizeFullMesh(mesh)
    
    visual = ColorVis(np.vstack([curvatures, curvatures, curvatures, np.ones_like(curvatures) ]).T)

    mesh.visual = visual
    with open(outputPath, 'wb+') as outfile:
        outfile.write( tm.exchange.ply.export_ply(mesh, encoding='binary', include_attributes=False) )
    

    
