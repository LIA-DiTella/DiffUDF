from src.submesh import preprocessMesh
import subprocess
import open3d as o3d
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess triangle mesh for training')
    parser.add_argument('input_path', metavar='path/to/mesh', type=str,
                        help='path to input mesh')
    parser.add_argument('output_folder', metavar='path/to/output/', type=str,
                        help='path to output folder')
    parser.add_argument('-a', '--alpha', type=float, default=7.5, help='order of distance between joints')
    parser.add_argument('-b', '--beta', type=int, default=15, help='minimum distance between joints')
    parser.add_argument('-s', '--std', type=float, default=6, help='standard diviation for mesh code')
    parser.add_argument('-f', '--full', action='store_true', help='leave mesh unfragmented')

    args = parser.parse_args()

    inputPath = args.input_path
    outputFolder = args.output_folder if args.output_folder[-1] == '/' else args.output_folder + '/'
    fileName = inputPath[ inputPath.rfind('/') + 1 : ]
    fileName = fileName[:fileName.rfind('.')]

    print('Extracting skeleton...')

    extractionProc = subprocess.run(["src/skeletonize/build/Skel", inputPath, outputFolder + fileName], capture_output=True)
    
    print( '    ' + extractionProc.stdout.decode(encoding='ascii').replace('\n', '\n' + '    ') )

    print('Preparing dataset...')
    print(f'    using alpha = {args.alpha}, beta = {args.beta}, std = {args.std}')
    print('    Saved to path: ', preprocessMesh( outputFolder + fileName, inputPath, outputFolder + fileName + '.txt', outputFolder + fileName + '_corr.txt', args.alpha, args.beta, args.std, args.full ))
    



