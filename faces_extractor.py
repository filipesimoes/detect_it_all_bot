from faces_util import detect_faces
import util
import cv2 as cv
import os
import numpy as np
from os.path import abspath


def main():
    parser = util.HelperParser(
        description='OpenCV faces utilities for extracting face from image files.')
    parser.add_argument('-f', '--face_classifier', required=True,
                        help='The face classifier to be used.')
    parser.add_argument('-e', '--eye_classifier', required=True,
                        help='The eye classifier to be used.')
    parser.add_argument('-p', '--path', required=True,
                        help='The path to search for images.')
    parser.add_argument('-o', '--output', required=True,
                        help='The output path.')
    parser.add_argument('-w', '--face_width', default=256,
                        type=int, help='The extracted face width.')
    parser.add_argument('-t', '--face_height', default=256,
                        type=int, help='The extracted face height.')
    parser.add_argument('-s', '--suffixes', nargs="+", default=".png .jpg .jpeg",
                        help="The image suffixes list. Default '.png .jpg .jpeg'")

    args = parser.parse_args()

    face_detector = cv.CascadeClassifier(args.face_classifier)
    eye_detector = cv.CascadeClassifier(args.eye_classifier)
    output = abspath(args.output)

    faces_ids = {}
    faces_names = {}
    faces_samples = []
    samples_ids = []
    count = 0
    files_count = 0
    for file in os.listdir(abspath(args.path)):
        file = abspath(args.path) + "/" + file
        extension = os.path.splitext(file)[1]
        if extension in args.suffixes:
            print(f"Loading file '{file}'...", end='')
            img = cv.imread(file)
            if (img is None):
                print(f"Could not load '{file}'.")
            else:
                faces = detect_faces(img, face_detector, eye_detector,
                                     desired_face_width=args.face_width,
                                     desired_face_height=args.face_height,
                                     rotations=[-45.0, -22.5, 0.0, 22.5, 45.0])
                print(f"{len(faces)} faces found.")
                for face in faces:
                    mat = face.mat
                    w = len(mat[0])
                    h = len(mat)
                    to_be_shown = img.copy()
                    to_be_shown[0:h, 0: w] = mat
                    (x, y, w, h) = face.bbox
                    cv.rectangle(to_be_shown, (x, y),
                                 (x + w, y + h), (255, 0, 0), 2)
                    for eye in face.eyes:
                        (xe, ye, we, he) = eye
                        cv.rectangle(to_be_shown, (x + xe, y + ye),
                                     (x + xe + we, y + ye + he), (0, 255, 0), 2)

                    cv.imshow('faces', to_be_shown)
                    cv.waitKey(200)
                    msg = """
Whose face is this?
Type a identifier for this face. For example: "john" or "manuel_batista". Don't use spaces!
                    """
                    face_id = None
                    face_name_detected = None
                    face_found = face.mat
                    gray_face = cv.cvtColor(face_found, cv.COLOR_BGR2GRAY)
                    if len(faces_ids) > 0:
                        face_id, confidence = recognizer.predict(
                            gray_face)
                        if face_id in faces_ids and confidence <= 50:
                            face_name_detected = faces_ids[face_id]
                            msg = msg + \
                                f"\n I think this is {face_name_detected} ({(100 - confidence):0.1f}%). Type <return> to confirm."

                    msg = msg + "\n To ignore this face type 'ignore'."
                    msg = msg + "\n Face identifier:"
                    face_name = input(msg).strip()
                    if face_name == "" and face_name_detected is not None:
                        face_name = face_name_detected
                    if face_name != "ignore":
                        if face_name in faces_names:
                            face_id = faces_names[face_name]
                        else:
                            face_id = count
                            faces_ids[face_id] = face_name
                            faces_names[face_name] = face_id
                            count = count + 1
                        faces_samples.append(gray_face)
                        samples_ids.append(face_id)
                        recognizer = cv.face.LBPHFaceRecognizer_create()
                        recognizer.train(
                            faces_samples, np.array([samples_ids]))
                        files_count = files_count + 1
                        cv.imwrite(
                            f"{output}/{files_count}.{face_id}.{face_name}.png", face_found)
                    cv.destroyAllWindows()
    if len(faces_ids) > 0:
        recognizer.write(output + "/faces.yml")
        with open(output + "/names.txt", 'w') as f:
            for face_id, face_name in faces_ids.items():
                f.write(f"{face_id},{face_name}\n")


if __name__ == "__main__":
    main()
