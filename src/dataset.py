import math
import numpy as np
import open3d as o3d
import open3d.core as o3c
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.utils.data import IterableDataset
import json

def readJson( path: str ):
    # Reading the PLY file with curvature info
    meshes = []
    means = []
    covs = []
    features = None

    with open(path, "r") as jsonFile:
        data = json.load( jsonFile )

        for joint in data['joints']:
            
            if features is None:
                features = len(np.array(joint['mean']))

            means.append( torch.from_numpy( np.array(joint['mean']) ))
            covs.append( torch.from_numpy( np.array(joint['cov']) ))     

            device = o3c.Device('CPU:0')
            mesh = o3d.t.geometry.TriangleMesh(device)
            mesh.vertex["positions"] = o3c.Tensor(np.asarray(joint['vertices']), dtype=o3c.float32)
            mesh.triangle["indices"] = o3c.Tensor(np.asarray(joint['triangles']), dtype=o3c.int32)
            mesh.vertex["normals"] = o3c.Tensor(np.asarray(joint['normals']), dtype=o3c.float32)
            mesh.vertex["curvature"] = o3c.Tensor(np.asarray(joint['curvature']), dtype=o3c.float32)
            meshes.append( mesh )
        
    return features, meshes, means, covs

def getCurvatureBins(curvatures: torch.Tensor, percentiles: list) -> list:
    allBins = []
    for curvature in curvatures:
        q = torch.quantile(curvature, torch.Tensor(percentiles))
        bins = [curvature.min().item(), curvature.max().item()]
        # Hack to insert elements of a list inside another list.
        bins[1:1] = q.data.tolist()
        allBins.append(bins)

    return allBins

def sampleTrainingData(
        meshes: list,
        samplesOnSurface: int,
        samplesFarSurface: int,
        samplesNearSurface: int,
        scenes: list,
        distributions: list,
        onSurfaceExceptions: list = [],
        domainBounds: tuple = ([-1, -1, -1], [1, 1, 1]),
        curvatureFractions: list = [],
        curvatureBins: list = [],
        nearSurfaceStd: float=0.01
):
        ## samples on surface
    surfacePoints = torch.cat([
        pointSegmentationByCurvature(
        mesh,
        samplesOnSurface,
        bins,
        curvatureFractions,
        exceptions
    ) for mesh, exceptions, bins in zip(meshes, onSurfaceExceptions, curvatureBins)], dim=0)

    surfacePoints = torch.from_numpy(surfacePoints.numpy())

        ## samples uniformly in domain (far)
    farDomainPoints = [ o3c.Tensor(np.random.uniform(
        domainBounds[0], domainBounds[1],
        (samplesFarSurface, 3)
    ), dtype=o3c.Dtype.Float32) for _ in range(len(meshes))]

    farDomainSDFs = torch.cat( [ torch.from_numpy(scene.compute_distance(points).numpy()) for scene, points in zip(scenes, farDomainPoints)], dim=0)
    farDomainPoints = torch.cat( [ torch.from_numpy(points.numpy()) for points in farDomainPoints ] )

        ## samples near surface
    displacement = MultivariateNormal(torch.Tensor([0,0,0]), torch.eye(3)*nearSurfaceStd )
    nearDomainPoints = [
        pointSegmentationByCurvature(
        mesh,
        samplesNearSurface,
        bins,
        curvatureFractions,
        exceptions
    )[..., :3] + displacement.sample( torch.Size([samplesNearSurface]) ) for mesh, exceptions, bins in zip(meshes, onSurfaceExceptions, curvatureBins)]
    
    nearDomainSDFs = torch.cat( [ torch.from_numpy(scene.compute_distance(o3c.Tensor(points.numpy())).numpy()) for scene, points in zip(scenes, nearDomainPoints)], dim=0)
    nearDomainPoints = torch.cat(nearDomainPoints, dim=0)

    domainNormals = torch.zeros((samplesNearSurface + samplesFarSurface, 3))

    # full dataset:
    fullSamples = torch.row_stack((
        surfacePoints[..., :3],
        farDomainPoints,
        nearDomainPoints
    ))
    fullNormals = torch.row_stack((
        surfacePoints[..., 3:6],
        domainNormals
    ))
    fullSDFs = torch.cat((
        torch.zeros(len(surfacePoints)),
        farDomainSDFs,
        nearDomainSDFs
    )).unsqueeze(1)
    fullCurvatures = torch.cat((
        surfacePoints[..., -1],
        torch.zeros(samplesNearSurface + samplesFarSurface)
    )).unsqueeze(1)
    
    fullOnSurfDistances = torch.cat( [
        torch.abs( dist.sample( torch.Size([ samplesOnSurface ]) )) for dist in distributions
    ] )

    fullFarSurfDistances = torch.cat( [
        torch.abs( dist.sample( torch.Size([ samplesFarSurface ]) )) for dist in distributions
    ] )

    fullNearSurfDistances = torch.cat( [
        torch.abs( dist.sample( torch.Size([ samplesNearSurface ]) )) for dist in distributions
    ] )

    fullDistances = torch.cat( (fullOnSurfDistances, fullFarSurfDistances, fullNearSurfDistances))

    return torch.column_stack( [fullDistances, fullSamples]).float().unsqueeze(0), fullNormals.float().unsqueeze(0), fullSDFs.float().unsqueeze(0), fullCurvatures.float().unsqueeze(0)

def pointSegmentationByCurvature(
        mesh: o3d.t.geometry.TriangleMesh,
        amountOfSamples: int,
        binEdges: np.array,
        proportions: np.array,
        exceptions: list = []
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

    if exceptions:
        index = torch.Tensor(
            list(set(range(pointsOnSurface.shape[0])) - set(exceptions)),
        ).int()
        pointsOnSurface = torch.index_select(
            pointsOnSurface, dim=0, index=index
        )

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
    def __init__(self, jsonPath: str,
                 batchSize: int,
                 samplingPercentiles: list,
                 batchesPerEpoch : int,
                 curvatureFractions: list = [],
                 curvaturePercentiles: list = []):
        super().__init__()

        print(f"Loading meshes \"{jsonPath}\".")
        self.features, self.meshes, self.means, self.covs = readJson(jsonPath)
        
        self.batchSize = batchSize
        self.samplesOnSurface = int(self.batchSize * samplingPercentiles[0])
        self.samplesFarSurface = int(self.batchSize * samplingPercentiles[1])
        self.samplesNearSurface = self.batchSize - self.samplesOnSurface - self.samplesFarSurface
        
        print(f"Fetching {self.samplesOnSurface} on-surface points per iteration.")
        print(f"Fetching {self.samplesFarSurface} far from surface points per iteration.")
        print(f"Fetching {self.samplesNearSurface} near surface points per iteration.")

        self.batchesPerEpoch = batchesPerEpoch

        print("Creating point-cloud and acceleration structures.")
        self.scenes = []
        for mesh in self.meshes:
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(mesh)
            self.scenes.append( scene )

        self.curvatureFractions = curvatureFractions

        self.curvatureBins = getCurvatureBins(
            [ torch.from_numpy(mesh.vertex["curvature"].numpy()) for mesh in self.meshes ],
            curvaturePercentiles
            )
        
        
    def __iter__(self):
        for _ in range(self.batchesPerEpoch):
            yield sampleTrainingData(
                meshes=self.meshes,
                samplesOnSurface=self.samplesOnSurface,
                samplesFarSurface=self.samplesFarSurface,
                samplesNearSurface=self.samplesNearSurface,
                scenes=self.scenes,
                curvatureFractions=self.curvatureFractions,
                curvatureBins=self.curvatureBins,
                distributions= [ MultivariateNormal( mean, cov ) if mean.shape[0] > 1 else Normal( mean, cov ) for mean, cov in zip(self.means, self.covs)],
                onSurfaceExceptions= [[] for _ in range(len(self.meshes))]
            )
    
if __name__ == "__main__":
    p = PointCloud(
        "results/juguete/juguete.json", batchSize=10, batchesPerEpoch=1,
        curvatureFractions=(0.2, 0.7, 0.1), curvaturePercentiles=(0.7, 0.85)
    )

    print(next(iter(p)))