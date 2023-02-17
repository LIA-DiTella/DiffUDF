from submesh.submesh import SkeletonMesh
import subprocess
import open3d as o3d
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess triangle mesh for training')
    parser.add_argument('input_path', metavar='path/to/mesh', type=str,
                        help='path to input mesh')
    parser.add_argument('output_folder', metavar='path/to/output/', type=str,
                        help='path to output folder')
    parser.add_argument('-s', '--save', action='store_true', help='save submeshes as *.ply')
    parser.add_argument('-p', '--ppu', type=float, default=5, help='density of joints per unit arc length of skeleton curve')
    parser.add_argument('-b', '--btol', type=float, default=0.5, help='branching joints tolerance')
    parser.add_argument('-n', '--nn', action='store_false', help='skip normalization and centering step')

    args = parser.parse_args()

    inputPath = args.input_path
    outputFolder = args.output_folder
    fileName = inputPath[ inputPath.rfind('/') + 1 : ]
    fileName = fileName[:fileName.rfind('.')]

    print('Extracting skeleton...')

    extractionProc = subprocess.run(["src/skeletonize/build/Skel", inputPath, outputFolder + fileName], capture_output=True)
    
    print( '    ' + extractionProc.stdout.decode(encoding='ascii').replace('\n', '\n' + '    ') )

    print('Sampling vertices and submeshing...')

    skel = SkeletonMesh(inputPath, 
                        outputFolder + fileName + '.txt',
                        outputFolder + fileName + '_corr.txt')
    
    skel.submesh(pointsPerUnit=args.ppu)

    print('    Joints sampled: ' + str(skel.amountOfJoints) )

    print('\nCleaning submeshes...')

    skel.postprocess(alpha=args.btol)

    if args.nn:
        print('\nNormalizing and centering submeshes...')
        skel.centerAndNormalize()

    if args.save:
        print('\nSaving submeshes...')
        for idx, (_, submesh, _) in enumerate(skel.getSubmeshes()):
            o3d.io.write_triangle_mesh(f'{outputFolder}{fileName}_{idx}.ply', submesh)

    print('\nSaving to json...')
    
    print('    Saved to path: ', skel.saveToJson(outputFolder + fileName + '.json'))
    



