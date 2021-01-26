import util
import openface_util
import cv2 as cv
from faces_util import detect_faces
import re
from os.path import abspath
import os


def main():
    parser = util.HelperParser(
        description='Openface representation utility.')
    parser.add_argument('-f', '--face_classifier', required=True,
                        help='The face classifier to be used.')
    parser.add_argument('-e', '--eye_classifier', required=True,
                        help='The eye classifier to be used.')
    parser.add_argument('-t', '--torch_face_model', required=True,
                        help='The torch network model to be used.')
    parser.add_argument('-p', '--path', required=True,
                        help='The path to search for images.')
    parser.add_argument('-o', '--output', required=True,
                        help='The output path.')
    parser.add_argument('-s', '--suffixes', nargs="+", default=".png .jpg .jpeg",
                        help="The image suffixes list. Default '.png .jpg .jpeg'")
    parser.add_argument('-d', '--img_dim', type=int, default=96,
                        help='The face dimension.')
    parser.add_argument('-c', '--cuda', type=bool, default=False,
                        help='CUDA')
    parser.add_argument('-a', '--openface_dir',
                        default="/usr/local/lib/python3.9/dist-packages/openface",
                        help='The openface instalation dir.')
    parser.add_argument('-r', '--regex', default="\\d+\\.\\d+\\.([A-Za-z]+)",
                        help='The regex to extract the face label.')

    args = parser.parse_args()
    face_detector = cv.CascadeClassifier(args.face_classifier)
    eye_detector = cv.CascadeClassifier(args.eye_classifier)
    net = openface_util.TorchNeuralNet(args.openface_dir,
                                       args.torch_face_model, imgDim=args.img_dim, cuda=args.cuda)
    output = abspath(args.output)
    label_regex = re.compile(args.regex)

    with open(output + "/representations.txt", 'w') as f:
        for file in os.listdir(abspath(args.path)):
            file = abspath(args.path) + "/" + file
            file_components = os.path.splitext(file)
            file_name = file_components[0]
            extension = file_components[1]
            label_match = label_regex.search(file_name)
            if extension in args.suffixes and label_match:
                label = label_match[1]
                print(f"Loading file '{file}' with label {label}.")
                img = cv.imread(file)
                rep = net.forward(img)
                rep_str = ';'.join(str(v) for v in rep)
                f.write(f"{file};{label};{rep_str}\n")


if __name__ == "__main__":
    main()
