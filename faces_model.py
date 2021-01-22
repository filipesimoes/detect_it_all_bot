from faces_util import align_face
import util
import cv2 as cv
import os
import numpy as np
from os.path import abspath

def main():
    parser = util.HelperParser(
        description='OpenCV faces utilities using cascade detector.')
    parser.add_argument('-f', '--face_classifier', required=True,
                        help='The face classifier to be used.')
    parser.add_argument('-e', '--eye_classifier', required=True,
                        help='The eye classifier to be used.')
    parser.add_argument('-p', '--path', required=True,
                        help='The path to search for images.')
    parser.add_argument('-o', '--output', required=True,
                        help='The model output path.')
    parser.add_argument('-s', '--suffixes', nargs="+", default=".png .jpg .jpeg",
                        help="The image suffixes list. Default '.png .jpg .jpeg'")

    args = parser.parse_args()

    face_detector = cv.CascadeClassifier(args.face_classifier)
    eye_detector = cv.CascadeClassifier(args.eye_classifier)
    recognizer = cv.face.LBPHFaceRecognizer_create()
    
    faces_ids = {}
    faces_names = {}
    count = -1
    for file in os.listdir(abspath(args.path)):
        file = abspath(args.path) + "/" + file
        extension = os.path.splitext(file)[1]
        if extension in args.suffixes:
            print(f"Loading file '{file}'.")
            img = cv.imread(file)
            if (img is None):
                print(f"Could not load '{file}'.")
            else:
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.3, 5)
                print(f"{len(faces)} faces detected in {file}.")
                for face_shape in faces:
                    (x, y, w, h) = face_shape
                    face = gray[y:y+h, x:x+w]
                    eyes = eye_detector.detectMultiScale(face, 1.3, 5)
                    if len(eyes) == 2:
                        face_found = align_face(gray, face_shape, eyes)
                        cv.imshow('face', face)
                        cv.imshow('face_found', face_found)
                        cv.waitKey(100)
                        msg = """
Whose face is this?
Type a identifier for this face. For example: "john" or "manuel_batista". Don't use spaces!
                        """
                        face_id = None
                        face_name_detected = None
                        if len(faces_ids) > 0:
                            face_id, confidence = recognizer.predict(face_found)
                            if face_id in faces_ids:
                                face_name_detected = faces_ids[face_id]
                                msg = msg + f"\n I think this is {face_name_detected} ({(100 - confidence):0.1f}%). Type <return> to confirm."

                        msg = msg + "\n To ignore this face type 'ignore'."
                        msg = msg + "\n Face identifier:"
                        face_name = input(msg).strip()
                        if face_name == "" and face_name_detected is not None:
                            face_name = face_name_detected
                        if face_name != "ignore":
                            if face_name in faces_names:
                                face_id = faces_names[face_name]
                            else:
                                face_id = count + 1
                                faces_ids[face_id] = face_name
                                faces_names[face_name] = face_id
                            recognizer.train([face_found], np.array([face_id]))
                    cv.destroyAllWindows()
    output = abspath(args.output)
    recognizer.write(output + "/faces.yml")
    with open(output + "/names.txt", 'w') as f:
        for face_id, face_name in faces_ids.items():
            f.write(f"{face_id},{face_name}")



if __name__ == "__main__":
    main()
