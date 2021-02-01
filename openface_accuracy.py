import util
import openface_util
import cv2 as cv
from os.path import abspath
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import os
import re
import time


def main():
    parser = util.HelperParser(
        description='Openface accuracy test.')
    parser = util.HelperParser(
        description='Openface representation utility.')
    parser.add_argument('-t', '--torch_face_model', required=True,
                        help='The torch network model to be used.')
    parser.add_argument('-p', '--path', required=True,
                        help='The path to search for images.')
    parser.add_argument('-e', '--test_path', required=True,
                        help='The path to search for test images.')
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

    label_regex = re.compile(args.regex)
    net = openface_util.TorchNeuralNet(args.openface_dir,
                                       args.torch_face_model, imgDim=args.img_dim, cuda=args.cuda)
    norm = Normalizer(norm='l2')
    try:
        (labels, reps) = get_representations(args.path,
                                             label_regex,
                                             args.suffixes,
                                             net,
                                             norm)
        (test_labels, test_reps) = get_representations(args.test_path,
                                                       label_regex,
                                                       args.suffixes,
                                                       net,
                                                       norm)
        le = LabelEncoder()
        le.fit(labels)
        labels_num = le.transform(labels)
        test_labels_num = le.transform(test_labels)

        model = SVC(kernel='linear', probability=True)
        model.fit(reps, labels_num)


        # predict
        yhat_train = model.predict(reps)
        yhat_test = model.predict(test_reps)
        # score
        score_train = accuracy_score(labels_num, yhat_train)
        score_test = accuracy_score(test_labels_num, yhat_test)
        # summarize
        print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))
    finally:
        net.close()


def get_representations(path: str,
                        label_regex,
                        suffixes,
                        net: openface_util.TorchNeuralNet,
                        norm: Normalizer):

    labels = []
    reps = []
    for file in os.listdir(abspath(path)):
        file = abspath(path) + "/" + file
        file_components = os.path.splitext(file)
        file_name = file_components[0]
        extension = file_components[1]
        label_match = label_regex.search(file_name)
        if extension in suffixes and label_match:
            label = label_match[1]
            img = cv.imread(file)
            t0 = time.time()
            rep = net.forward(img)
            took = time.time() - t0
            rep = norm.transform(rep.reshape(1, -1))[0]
            print(f"Analising file '{file}' with label {label} took {took:.01f}s.")
            labels.append(label)
            reps.append(rep)

    return (labels, reps)


if __name__ == "__main__":
    main()
