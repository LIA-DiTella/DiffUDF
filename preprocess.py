from src.preprocess_mesh import preprocessMesh
import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess triangle mesh for training')
    parser.add_argument('input_path', metavar='path/to/mesh', type=str,
                        help='path to input mesh')
    parser.add_argument('output_path', metavar='path/to/output/point_cloud', type=str,
                        help='path to output point cloud')
    parser.add_argument('-nn', '--not_normalize', action='store_true', help='skip normalization step')
    parser.add_argument('-s', '--sub_it', type=int, default=0, help='loop subdivide iterations')

    args = parser.parse_args()

    inputPath = args.input_path
    outputPath = args.output_path

    if os.path.isfile( inputPath ):
        print('Preparing point cloud...')
        preprocessMesh( 
                outputPath, 
                inputPath,
                not_normalize=args.not_normalize )
    else:
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(inputPath)):
            for file in filenames:
                if file[-4:] == '.obj':
                    print(f'Processing {i}-th mesh...')
                    preprocessMesh( 
                        os.path.join(dirpath, file[:-4] + '_p.ply'), 
                        os.path.join(os.path.join(dirpath, file)), 
                        not_normalize=args.not_normalize, 
                        subdivide=args.sub_it 
                    )
    



