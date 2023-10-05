import numpy as np
import trimesh as tm
import igl

def calculateCurvature( v, f, subdivide ):
    sv, sf = igl.loop(v,f, subdivide)

    if subdivide > 0:
        print(f'Upsampled mesh from {len(v)} to {len(sv)} vertices.')
        
    print('Computing curvatures')
    pd1, pd2, pk1, pk2 = igl.principal_curvature(sv, sf)

    return sv, igl.per_vertex_normals( sv, sf ),sf, (pk1 + pk2)/2

def normalizeFullMesh( mesh ):
    T = np.block( [ [np.eye(3,3), -1 * mesh.center_mass.reshape((3,1))], [np.eye(1,4,k=3)]])

    mesh.apply_transform( T )

    max_coord = np.max( np.abs( mesh.vertices ))

    S = np.block([ [np.eye(3,3) * (1 / (max_coord + max_coord * 0.1)), np.zeros((3,1))], [np.eye(1,4,k=3)] ]) 
    mesh.apply_transform( S )

    return S @ T

class ColorVis:
    def __init__(self, colors):
        self.vertex_colors = np.uint8( colors * 255 )
        self.kind = 'vertex'

def preprocessMesh( outputPath, meshFile, not_normalize=True, subdivide=0 ):
    if meshFile[-4:] == '.obj':
        vertices,_,normals, faces,_,_ = igl.read_obj(meshFile)
    
    subdiv_vertices, subdiv_normals, subdiv_triangles, curvatures = calculateCurvature( vertices, faces, subdivide=subdivide )
    # elimino outliers
    curvatures = np.clip( curvatures, np.percentile(curvatures, 5), np.percentile(curvatures, 95) )

    # normalizo curvaturas
    curvatures -= np.min(curvatures)
    curvatures /= np.max(curvatures)

    normals_norm = np.linalg.norm( subdiv_normals, axis=-1 )
    mask_invalid_normals = np.isclose( normals_norm, np.zeros_like(normals_norm) )
    norm_factor = np.where(
        mask_invalid_normals,
        np.ones_like(normals_norm),
        normals_norm
    )
    normalized_normals = subdiv_normals / np.vstack( [norm_factor, norm_factor, norm_factor] ).T 
    mesh = tm.Trimesh( subdiv_vertices, subdiv_triangles, vertex_normals=normalized_normals )
    mesh.update_vertices( np.logical_not(mask_invalid_normals) )

    print(mesh.vertex_normals.shape)

    if not not_normalize:
        print('Normalizing')
        transform_matrix = normalizeFullMesh(mesh)

        transformed_gt_mesh = tm.Trimesh( vertices, faces, vertex_normals=normalized_normals )
        transformed_gt_mesh.apply_transform( transform_matrix )

        transformed_gt_mesh.export( meshFile[:meshFile.find('.')] + '_t.obj' )
    
    valid_curvatures = curvatures[np.logical_not(mask_invalid_normals)]
    visual = ColorVis(np.vstack([valid_curvatures, valid_curvatures, valid_curvatures, np.ones_like(valid_curvatures) ]).T)

    mesh.visual = visual
    with open(outputPath, 'wb+') as outfile:
        outfile.write( tm.exchange.ply.export_ply(mesh, encoding='binary', include_attributes=False) )
    

    
