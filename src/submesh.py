import numpy as np
import open3d as o3d
import pymeshlab as pm

def calculateCurvature( meshFile, num_vertices ):
    pyMeshset = pm.MeshSet()
    pyMeshset.load_new_mesh(meshFile)
    pyMesh = pyMeshset.current_mesh()

    d = pyMeshset.apply_filter("compute_scalar_by_discrete_curvature_per_vertex", curvaturetype='Mean Curvature')
    pyMeshset.compute_new_custom_scalar_attribute_per_vertex(name="v_curv", expr="q")
    v_curv = pyMesh.vertex_custom_scalar_attribute_array('v_curv')
    
    if len(v_curv) != num_vertices:
        print(len(v_curv) , num_vertices)
        raise ValueError('The mesh has repeated vertices')
    
    return np.clip( v_curv, a_min = -1 * float(d['90_percentile']), a_max = float(d['90_percentile']))


def normalizeFullMesh( mesh ):
    T = np.block( [ [np.eye(3,3), -1 * mesh.get_center().numpy().reshape((3,1))], [np.eye(1,4,k=3)]])

    mesh.transform( o3d.core.Tensor(T) )

    max_coord = np.max( np.abs( mesh.vertex.positions.numpy()))

    S = np.block([ [np.eye(3,3) * (1 / (max_coord + max_coord * 0.1)), np.zeros((3,1))], [np.eye(1,4,k=3)] ]) 
    mesh.transform( o3d.core.Tensor(S) )

def preprocessMesh( outputPath, meshFile, not_normalize=True ):
    mesh = o3d.t.io.read_triangle_mesh( meshFile )

    print('Computing curvatures')
    curvatures = calculateCurvature( meshFile, len(mesh.vertex.positions.numpy()) )
    curvatures -= np.min(curvatures)
    curvatures /= np.max(curvatures)
    mesh.vertex.colors = o3d.core.Tensor( np.vstack([curvatures, curvatures, curvatures]).T )
    
    mesh.compute_vertex_normals( normalized=True )

    if not not_normalize:
        print('Normalizing')
        normalizeFullMesh(mesh)

    o3d.t.io.write_triangle_mesh( outputPath, mesh )
    

    
