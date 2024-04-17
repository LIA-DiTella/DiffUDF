# DUDF: Differentiable Unsigned Distance Fields with Hyperbolic Scaling
Miguel Fainstein $^{\text{1}}$, Viviana Siless $^{\text{1}}$, Emmanuel Iarussi $^{\text{1},\text{2}}$ 

$^{\text{1}}$ Universidad Torcuato Di Tella $^{\text{2}}$ CONICET

Repository for the CVPR 2024 [paper](https://lia-ditella.github.io/DUDF/).

<img src="resources/teaser.png">

## Citation

If you find our project useful, please cite the following

```     
@misc{
    fainstein2024dudf,
    title={DUDF: Differentiable Unsigned Distance Fields with Hyperbolic Scaling}, 
    author={Miguel Fainstein and Viviana Siless and Emmanuel Iarussi},
    year={2024},
    eprint={2402.08876},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Install

The project requires a Linux machine and GPU(s) with CUDA 11.7
Other architectures are possible (eg. full CPU) but small changes in the code base might be necessary.

Code written in Python 3.10, main libraries used are *pytorch*, *numpy*, *pandas*, *open3d*, *trimesh*, *pytorch3d* and *matplotlib*.
For a quick installation run:

```
conda create -n dudf -f dudf.yml
conda activate dudf
```

If you want to utilize [MeshUDF](https://bguillard.github.io/meshudf/)'s marching cube method (mentioned in the paper as MC2), run the following:
```
cd src/marching_cubes
python setup.py build_ext --inplace
```

## Preprocessing
To preprocess a triangle mesh (or directory with triangle meshes), run
```
python preprocess.py input/mesh output/folder -s {NUMBER_OF_SAMPLES}
```

where NUMBER_OF_SAMPLES is the amount of surface points used for training. In our experiments we utilized 100k for simpler shapes and 200k for more complex shapes.

This generates two files, *mesh_file_t.obj* and *mesh_file_pc.ply*. The former is the original mesh transformed to fit the cube of side length 2, and the latter is the sampled point cloud.

## Training

For training, run
```
python train.py {PATH/CONFIG/FILE} {DEVICE}
```
An example configuration file can be found in *configs/train_cfg.json*. Parameter *device* is the number of the CUDA GPU to utilize. Training finishes by generating a 2D slice image of the level sets of the function and the gradient norm in the same slice; this was utilized during experimentation to analize the learned field. Also two marching cubes reconstructions are computed.

## Rendering

#### Sphere tracing
To render through means of sphere tracing algorithm, run
```
python generate_st.py {PATH/CONFIG/FILE}
```

An example configuration file can be found in *configs/st_cfg.json*. To render curvatures choose parameter *plot_curvatures* to be either *mean* or *gaussian*. Additionally, parameter *reflection_method* allows for two different illumination algorithms *ward* or *blinn-phong*.

#### Marching cubes

To render through means of gradient-based marching cubes algorithms, run
```
python generate_mc.py {PATH/CONFIG/FILE}
```

An example configuration file can be found in *configs/mc_cfg.json*. It is possible to select the version of the algorithm, either *cap*, *meshudf* or *both*; in reference to the methods described in papers [CAP-UDF](https://junshengzhou.github.io/CAP-UDF/) and [MeshUDF](https://bguillard.github.io/meshudf/).

#### Point cloud extraction

Lastly, we provide source code to perform point cloud extraction following the scheme laid in [NDF](https://virtualhumans.mpi-inf.mpg.de/ndf/). However, our method allows to not only extract a dense point cloud, but the normal field associated with it. Eventhough normal fields of open surfaces are not always orientable, we try to orient these utilizing library *Open3D*. If the original mesh was a closed surface, it can be reconstructed by means of *Poisson screening* (Kazhdan, 2013) method, which has much better results than the proposed *Ball Pivoting* approach. Simply, run

```
python generate_pc.py {PATH/CONFIG/FILE}
```

An example configuration file can be found in *configs/pc_cfg.json*.

## Example

As an example we can perform the full pipeline on the provided *beetle* mesh by running:
```
python preprocess.py data/beetle/beetle.obj data/beetle -s 100000
```
Then we can train a neural network:
```
python train.py configs/train_cfg.json 0
```
After training is over, in folder *results/beetle/experiment_1* we find all the files related with the process. Folder *reconstructions* will have two images:
1. *distance_fields.png* has information about the field and the gradient norm, with comparisons with ground truth values.
2. *pred_grad.png* has information about the gradient field plotted as a normal map.
Additionally, both marching cubes reconstructions will be available under the names *mc_mesh_best_CAP.obj* and *mc_mesh_best_MU.obj*. Again, here CAP(-UDF) and MU (MeshUDF) make reference to the papers where they were proposed.

After that we can perform sphere tracing reconstructions by running
```
python generate_st.py configs/st_cfg.json && python generate_st.py configs/st_mean_cfg.json
```
We can see the rendering with and without plotting mean curvatures.

Lastly, to extract the oriented point cloud run
```
python generate_pc.py configs/pc_cfg.json
```

## Contact

For questions and comments contact Miguel Fainstein via mail

## License

...