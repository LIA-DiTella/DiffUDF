import math
import numpy as np
import open3d as o3d
import open3d.core as o3c
import torch
from torch.utils.data import IterableDataset

def o3c_to_torch( tensor: o3c.Tensor ) -> torch.Tensor:
    return torch.utils.dlpack.from_dlpack( tensor.to_dlpack() )

def torch_to_o3c( tensor: torch.Tensor ) -> o3c.Tensor:
    return o3c.Tensor.from_dlpack( torch.utils.dlpack.to_dlpack(tensor) )

def sampleTrainingData(
        surface_pc: o3d.t.geometry.PointCloud,
        samplesOnSurface: int,
        samplesOffSurface: int,
        scene,
        domainBounds: tuple = ([-1, -1, -1], [1, 1, 1]),
):
    
    surfaceSamples = surface_pc.select_by_index( 
        o3c.Tensor.from_numpy( np.random.randint(0, len(surface_pc.point['positions']), samplesOnSurface) )
    )

    ## samples uniformly in domain
    samplesFar = samplesOffSurface // 2
    samplesNear = samplesOffSurface - samplesFar

    domainPoints = o3c.Tensor(np.random.uniform(
        domainBounds[0], domainBounds[1],
        (samplesFar, 3)
    ), dtype=o3c.Dtype.Float32)

    domainSDFs = torch.from_numpy(scene.compute_signed_distance(domainPoints).numpy())
    domainPoints = o3c_to_torch( domainPoints )

    surfacePointsSubset = surfaceSamples.select_by_index( 
        o3c.Tensor.from_numpy( np.random.randint(0, samplesOnSurface, (samplesNear,1)) )
    )

    surfacePointsSubsetNormals = o3c_to_torch( surfacePointsSubset.point['normals'] ).squeeze(1)
    surfacePointsSubset = o3c_to_torch( surfacePointsSubset.point['positions'] ).squeeze(1)

    surfaceNormals = o3c_to_torch( surfaceSamples.point['normals'] )
    surfacePoints = o3c_to_torch( surfaceSamples.point['positions'] )

    closePoints = ( surfacePointsSubset + surfacePointsSubsetNormals * torch.normal(0, 0.01, (samplesNear, 1) ) )

    closeSDFs = o3c_to_torch( scene.compute_signed_distance( torch_to_o3c(closePoints).to( o3c.Dtype.Float32 ) ) )

    domainNormals = torch.zeros((samplesOffSurface, 3))

    # full dataset:
    fullSamples = torch.row_stack((
        surfacePoints,
        domainPoints,
        closePoints
    ))
    fullNormals = torch.row_stack((
        surfaceNormals,
        domainNormals
    ))
    fullSDFs = torch.cat((
        torch.zeros(samplesOnSurface),
        domainSDFs,
        closeSDFs
    )).unsqueeze(1)

    return fullSamples.float().unsqueeze(0), fullNormals.float().unsqueeze(0), fullSDFs.float().unsqueeze(0)

def shortestDistance( P, X ):
    sqnormP = torch.sum( P * P, dim=1)
    sqnormX = torch.sum( X * X, dim=1)

    shDistances, _ = torch.min( sqnormX.repeat( P.shape[0], 1 ) - 2 * ( P @ X.T ), dim=1 )

    return torch.sqrt( shDistances + sqnormP )

def sampleTrainingDataPC(
        surface_pc: torch.Tensor,
        surface_normals: torch.Tensor,
        samplesOnSurface: int,
        samplesOffSurface: int,
        domainBounds: tuple = ([-1, -1, -1], [1, 1, 1]),
):
    device = surface_pc.get_device()
    
    surfaceIndexs = torch.from_numpy( np.random.randint(0, surface_pc.shape[0], samplesOnSurface) ).to( device )
    surfacePoints = surface_pc[ surfaceIndexs ]
    surfaceNormals = surface_normals[ surfaceIndexs ]

    ## samples uniformly in domain
    samplesFar = samplesOffSurface // 2
    samplesNear = samplesOffSurface - samplesFar

    domainPoints = np.random.uniform(
        domainBounds[0], domainBounds[1],
        (samplesFar, 3)
    )

    domainPoints = torch.from_numpy( domainPoints ).to( device )
    domainSDFs = shortestDistance( domainPoints, surface_pc )

    nearIndexs = torch.from_numpy( np.random.randint(0, samplesOnSurface, samplesNear ) ).to( device )
    surfacePointsSubset = surfacePoints[ nearIndexs ]
    surfacePointsSubsetNormals = surfaceNormals[ nearIndexs ]

    offset = torch.normal(0, 0.01, (samplesNear, 1) ).to( device )
    closePoints = ( surfacePointsSubset + surfacePointsSubsetNormals * offset )
    closeSDFs = torch.abs( offset )

    domainNormals = torch.zeros((samplesOffSurface, 3)).to(device)

    # full dataset:
    fullSamples = torch.row_stack((
        surfacePoints,
        domainPoints,
        closePoints
    ))
    fullNormals = torch.row_stack((
        surfaceNormals,
        domainNormals
    ))
    fullSDFs = torch.cat((
        torch.zeros(samplesOnSurface).to(device),
        domainSDFs,
        closeSDFs.squeeze(1)
    )).unsqueeze(1)

    return fullSamples.float().unsqueeze(0), fullNormals.float().unsqueeze(0), fullSDFs.float().unsqueeze(0)


class PointCloud(IterableDataset):
    def __init__(
                self, meshPath: str,
                batchSize: int,
                samplingPercentiles: list,
                batchesPerEpoch : int,
                device: torch.device,
                onlyPCloud=False,
                ):
        super().__init__()

        self.onlyPCloud = onlyPCloud

        print(f"Loading data \"{meshPath}\".")
        
        self.surface_pc = o3d.t.io.read_point_cloud(meshPath + '_pc.ply')
        
        if not self.onlyPCloud: 
            self.mesh = o3d.t.io.read_triangle_mesh(meshPath + '_t.obj')
            print("Creating point-cloud and acceleration structures.")
            self.scene = o3d.t.geometry.RaycastingScene()
            self.scene.add_triangles(self.mesh)
        else:
            self.surface_normals = o3c_to_torch( self.surface_pc.point.normals ).to( device )
            self.surface_pc = o3c_to_torch( self.surface_pc.point.positions ).to( device )


        self.batchSize = batchSize
        self.samplesOnSurface = int(self.batchSize * samplingPercentiles[0])
        self.samplesFarSurface = int(self.batchSize * samplingPercentiles[1])
        
        print(f"Fetching {self.samplesOnSurface} on-surface points per iteration.")
        print(f"Fetching {self.samplesFarSurface} far from surface points per iteration.")

        self.batchesPerEpoch = batchesPerEpoch

    def __iter__(self):
        for _ in range(self.batchesPerEpoch):
            if self.onlyPCloud:
                yield sampleTrainingDataPC(
                    surface_pc=self.surface_pc,
                    surface_normals=self.surface_normals,
                    samplesOnSurface=self.samplesOnSurface,
                    samplesOffSurface=self.samplesFarSurface,
                )
            else:
                yield sampleTrainingData(
                    surface_pc=self.surface_pc,
                    samplesOnSurface=self.samplesOnSurface,
                    samplesOffSurface=self.samplesFarSurface,
                    scene=self.scene
                )