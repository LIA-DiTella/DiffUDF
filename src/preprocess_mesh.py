import numpy as np
import open3d as o3d
import os

def normalizeMesh( mesh ):
    T = np.block( [ [np.eye(3,3), -1 * mesh.get_center().reshape((3,1))], [np.eye(1,4,k=3)]])

    mesh.transform( T )

    max_coord = np.max( np.abs( mesh.vertices ))

    S = np.block([ [np.eye(3,3) * (1 / (max_coord + max_coord * 0.1)), np.zeros((3,1))], [np.eye(1,4,k=3)] ]) 
    mesh.transform( S )

    return S @ T

def preprocessMesh( outputPath, meshFile, surfacePoints=1e5 ):
    mesh = o3d.io.read_triangle_mesh(meshFile)
    mesh.normalize_normals()
    normalizeMesh( mesh )

    mesh_name = meshFile[meshFile.rfind('/') + 1 : meshFile.rfind('.') ]
    print(mesh_name)

    o3d.io.write_triangle_mesh( os.path.join( outputPath , mesh_name + '_t.obj'), mesh )

    surfacePointCloud = mesh.sample_points_uniformly(number_of_points=int(surfacePoints), use_triangle_normal=True)
    o3d.io.write_point_cloud(  os.path.join( outputPath , mesh_name + '_pc.ply'), surfacePointCloud )
    
