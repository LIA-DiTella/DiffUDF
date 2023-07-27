from src.submesh import preprocessMesh
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess triangle mesh for training')
    parser.add_argument('input_path', metavar='path/to/mesh', type=str,
                        help='path to input mesh')
    parser.add_argument('output_path', metavar='path/to/output/point_cloud', type=str,
                        help='path to output point cloud')
    parser.add_argument('-nn', '--not_normalize', action='store_true', help='skip normalization step')

    args = parser.parse_args()

    inputPath = args.input_path
    outputPath = args.output_path

    print('Preparing point cloud...')
    preprocessMesh( 
              outputPath, 
              inputPath,
              not_normalize=args.not_normalize )
    



