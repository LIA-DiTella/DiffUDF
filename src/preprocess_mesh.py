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

def normalizePointCloud( pointCloud ):
    T = np.block( [ [np.eye(3,3), -1 * pointCloud.get_center().reshape((3,1))], [np.eye(1,4,k=3)]])

    pointCloud.transform( T )

    max_coord = np.max( np.abs( pointCloud.points ))

    S = np.block([ [np.eye(3,3) * (1 / (max_coord + max_coord * 0.1)), np.zeros((3,1))], [np.eye(1,4,k=3)] ]) 
    pointCloud.transform( S )

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
    
def preprocessPointCloud( outputPath, pcFile, surfacePoints=1e5 ):
    pointcloud = o3d.io.read_point_cloud( pcFile )
    pointcloud.normalize_normals()
    normalizePointCloud( pointcloud )

    pointcloud_name = pcFile[pcFile.rfind('/') + 1 : pcFile.rfind('.') ]
    print(pointcloud_name)

    points = np.asarray( pointcloud.points )
    normals = np.asarray( pointcloud.normals )

    if surfacePoints > len(points):
        raise ValueError(f'Cannot sample more points ({surfacePoints}) than present on the input pointcloud ({len(points)}).')
    
    indices = np.random.choice(len(points), size=surfacePoints, replace=False)

    sampled_points = points[indices]
    sampled_normals = normals[indices]

    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    sampled_pcd.normals = o3d.utility.Vector3dVector(sampled_normals)

    o3d.io.write_point_cloud( os.path.join( outputPath , pointcloud_name + '_t.ply'), pointcloud )
    o3d.io.write_point_cloud(  os.path.join( outputPath , pointcloud_name + '_pc.ply'), sampled_pcd )