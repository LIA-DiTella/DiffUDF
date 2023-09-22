import math
import numpy as np
import open3d as o3d
import open3d.core as o3c
import torch
from torch.utils.data import IterableDataset

def getCurvatureBins(curvature: torch.Tensor, percentiles: list) -> list:
    q = torch.quantile(curvature, torch.Tensor(percentiles))
    bins = [curvature.min().item(), curvature.max().item()]
    # Hack to insert elements of a list inside another list.
    bins[1:1] = q.data.tolist()

    return bins

def sampleTrainingData(
        mesh: list,
        samplesOnSurface: int,
        samplesOffSurface: int,
        scene,
        domainBounds: tuple = ([-1, -1, -1], [1, 1, 1]),
        curvatureFractions: list = [],
        curvatureBins: list = []
):
        ## samples on surface
    surfacePoints = pointSegmentationByCurvature(
        mesh,
        samplesOnSurface,
        curvatureBins,
        curvatureFractions
    )


    ## samples uniformly in domain (far)
    domainPoints = o3c.Tensor(np.random.uniform(
        domainBounds[0], domainBounds[1],
        (samplesOffSurface, 3)
    ), dtype=o3c.Dtype.Float32)

    domainSDFs = torch.from_numpy(scene.compute_distance(domainPoints).numpy())
    domainPoints = torch.from_numpy(domainPoints.numpy())

    domainNormals = torch.zeros((samplesOffSurface, 3))

    # full dataset:
    fullSamples = torch.row_stack((
        surfacePoints[..., :3],
        domainPoints
    ))
    fullNormals = torch.row_stack((
        surfacePoints[..., 3:6],
        domainNormals
    ))
    fullSDFs = torch.cat((
        torch.zeros(len(surfacePoints)),
        domainSDFs
    )).unsqueeze(1)

    return fullSamples.float().unsqueeze(0), fullNormals.float().unsqueeze(0), fullSDFs.float().unsqueeze(0)

def pointSegmentationByCurvature(
        mesh: o3d.t.geometry.TriangleMesh,
        amountOfSamples: int,
        binEdges: np.array,
        proportions: np.array
):
    
    def fillBin( points, curvatures, amountSamples, lowerBound, upperBound ):
        pointsInBounds = points[(curvatures >= lowerBound) & (curvatures <= upperBound), ...]
        maskSampledPoints = np.random.choice(
            range(pointsInBounds.shape[0]),
            size=amountSamples,
            replace=True if amountSamples > pointsInBounds.shape[0] else False
        )
        return pointsInBounds[maskSampledPoints, ...]

    pointsOnSurface = torch.column_stack((
        torch.from_numpy(mesh.vertex["positions"].numpy()),
        torch.from_numpy(mesh.vertex["normals"].numpy()),
        torch.from_numpy(mesh.vertex["curvature"].numpy())
    ))

    curvatures = pointsOnSurface[..., -1]

    pointsLowCurvature = fillBin( pointsOnSurface, curvatures, int(math.floor(proportions[0] * amountOfSamples)), binEdges[0], binEdges[1])
    pointsMedCurvature = fillBin( pointsOnSurface, curvatures, int(math.ceil(proportions[1] * amountOfSamples)), binEdges[1], binEdges[2])
    pointsHighCurvature = fillBin( pointsOnSurface, curvatures, amountOfSamples - pointsLowCurvature.shape[0] - pointsMedCurvature.shape[0] , binEdges[2], binEdges[3])

    return torch.cat((
        pointsLowCurvature,
        pointsMedCurvature,
        pointsHighCurvature
    ), dim=0)



class PointCloud(IterableDataset):
    def __init__(self, meshPath: str,
                 batchSize: int,
                 samplingPercentiles: list,
                 batchesPerEpoch : int,
                 curvatureFractions: list = [],
                 curvaturePercentiles: list = []):
        super().__init__()

        print(f"Loading mesh \"{meshPath}\".")

        self.mesh = o3d.t.io.read_triangle_mesh(meshPath)
        self.mesh.vertex.curvature = o3c.Tensor( np.expand_dims(self.mesh.vertex.colors.numpy()[:, 0], -1) )
        del self.mesh.vertex.colors

        self.batchSize = batchSize
        self.samplesOnSurface = int(self.batchSize * samplingPercentiles[0])
        self.samplesFarSurface = int(self.batchSize * samplingPercentiles[1])
        
        print(f"Fetching {self.samplesOnSurface} on-surface points per iteration.")
        print(f"Fetching {self.samplesFarSurface} far from surface points per iteration.")

        self.batchesPerEpoch = batchesPerEpoch

        print("Creating point-cloud and acceleration structures.")
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(self.mesh)

        self.curvatureFractions = curvatureFractions

        self.curvatureBins = getCurvatureBins( torch.from_numpy(self.mesh.vertex.curvature.numpy()), curvaturePercentiles)
        
    def __iter__(self):
        for _ in range(self.batchesPerEpoch):
            yield sampleTrainingData(
                mesh=self.mesh,
                samplesOnSurface=self.samplesOnSurface,
                samplesOffSurface=self.samplesFarSurface,
                scene=self.scene,
                curvatureFractions=self.curvatureFractions,
                curvatureBins=self.curvatureBins,
            )