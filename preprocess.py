from src.preprocess_mesh import preprocessMesh, preprocessPointCloud
import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess triangle mesh for training')
    parser.add_argument('input_path', metavar='path/to/mesh', type=str,
                        help='path to input mesh')
    parser.add_argument('output_path', metavar='path/to/output/folder/', type=str,
                        help='path to output point cloud')
    parser.add_argument('-s', '--samples', type=int, default=1e5, help='surface samples')
    parser.add_argument('-pc', '--pointcloud', action='store_true', help='use pointcloud as input w/o need of triangle mesh')

    args = parser.parse_args()

    inputPath = args.input_path
    outputPath = args.output_path

    if os.path.isfile( inputPath ):
        print('Preparing point cloud...')

        if args.pointcloud:
            preprocessPointCloud(
                outputPath, 
                inputPath,
                surfacePoints=args.samples 
            )
        else:
            preprocessMesh( 
                    outputPath, 
                    inputPath,
                    surfacePoints=args.samples )
    else:
        if args.pointcloud:
            for dirpath, dirnames, filenames in os.walk(inputPath):
                for file in filenames:
                    if file[-4:] == '.ply' and file[-6:-4] != '_t' and file[-7:-4] != '_pc':
                        print(f'Processing {dirpath[dirpath.rfind("/")+1:]}...')
                        preprocessPointCloud( 
                            #os.path.join(dirpath, file[:-4]), 
                            dirpath,
                            os.path.join(os.path.join(dirpath, file)), 
                            surfacePoints=args.samples 
                        )
        else:
            for dirpath, dirnames, filenames in os.walk(inputPath):
                for file in filenames:
                    if file[-4:] == '.obj' and file[-6:-4] != '_t' and file[-7:-4] != '_pc':
                        print(f'Processing {dirpath[dirpath.rfind("/")+1:]}...')
                        preprocessMesh( 
                            os.path.join(dirpath, file[:-4]), 
                            os.path.join(os.path.join(dirpath, file)), 
                            surfacePoints=args.samples 
                        )
    



