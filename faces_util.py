import util
import cv2 as cv
import numpy as np
from typing import NamedTuple, List, Tuple
from math import cos, sin


class Face(NamedTuple):
    bbox: Tuple = None
    mat: np.ndarray = None
    eyes: List[Tuple] = None


def detect_faces(img,
                 face_classifier: cv.CascadeClassifier,
                 eye_classifier: cv.CascadeClassifier,
                 scale_factor: float = 1.3,
                 min_neighbors: int = 5,
                 max_faces: int = None,
                 desired_left_eye=(0.32, 0.32),
                 desired_face_width=256,
                 desired_face_height=None,
                 desired_eye_center: float = 0.8,
                 rotations: List[float] = [0.0]) -> List[Face]:
    """
    Detect faces and align it based on eyes landmarks.
    """
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    detected = []
    for angle in rotations:
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        if angle == 0.0:
            rotated = img
        else:
            rotated = cv.warpAffine(img, M, (w, h))
        gray = cv.cvtColor(rotated, cv.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(
            gray, scale_factor, min_neighbors)
        for face_bbox in faces:
            (x, y, w, h) = face_bbox
            face = gray[y:y+h, x:x+w]
            eyes = eye_classifier.detectMultiScale(
                face, scale_factor, min_neighbors)
            if len(eyes) >= 2:
                face_aligned = align_face(
                    rotated, face_bbox, eyes,
                    desired_left_eye=desired_left_eye,
                    desired_face_width=desired_face_width,
                    desired_face_height=desired_face_height,
                    desired_eye_center=desired_eye_center,)
                #angle = -1 * angle
                #ix = int(cos(angle) * x - sin(angle) * y)
                #iy = int(sin(angle) * x + cos(angle) * y)
                #face_bbox = (ix, iy, w, h)
                detected.append(Face(
                    bbox=face_bbox,
                    mat=face_aligned,
                    eyes=eyes
                ))
                if max_faces is not None and len(detected) == max_faces:
                    return detected

    return detected


# Modified from https://github.com/jrosebr1/imutils/blob/master/imutils/face_utils/facealigner.py
def align_face(img,
               face_shape,
               eyes_shapes,
               desired_left_eye=(0.32, 0.32),
               desired_face_width: int = 256,
               desired_face_height: int = None,
               desired_eye_center: float = 0.8):

    if desired_face_height is None:
        desired_face_height = int(desired_face_width * 1.2)

    (left_eye_shape, right_eye_shape) = detect_eyes(eyes_shapes)

    (x, y, w, h) = face_shape
    (x_le, y_le, w_le, h_le) = left_eye_shape
    (x_re, y_re, w_re, h_re) = right_eye_shape

    left_eye_center_x = x + x_le + (w_le // 2)
    right_eye_center_x = x + x_re + (w_re // 2)

    left_eye_center_y = y + y_le + (h_le // 2)
    right_eye_center_y = y + y_re + (h_re // 2)

    dx = right_eye_center_x - left_eye_center_x
    dy = right_eye_center_y - left_eye_center_y
    angle = np.degrees(np.arctan2(dy, dx))

    dist = np.sqrt((dx ** 2) + (dy ** 2))
    desired_right_eye_x = 1.0 - desired_left_eye[0]
    desired_dist = (desired_right_eye_x - desired_left_eye[0])
    desired_dist *= desired_face_width
    scale = desired_dist / dist

    eyes_center = ((left_eye_center_x + right_eye_center_x) // 2,
                   (left_eye_center_y + right_eye_center_y) // 2)

    M = cv.getRotationMatrix2D(eyes_center, angle, scale)

    height = img.shape[0]
    width = img.shape[1]
    img = cv.warpAffine(img, M, (width, height), flags=cv.INTER_CUBIC)

    w = desired_face_width
    h = desired_face_height
    eye_center_x = eyes_center[0]
    eye_center_y = eyes_center[1]
    x = eye_center_x - desired_face_width // 2
    dy = int(desired_eye_center * desired_face_height) // 2
    y = eye_center_y - dy
    xf = x + w
    yf = y + h
    x = max(0, x)
    y = max(0, y)
    face = img[y:yf, x:xf]
    return face


def detect_eyes(eyes):
    assert len(eyes) >= 2
    eyes = sorted(eyes, key=lambda eye: eye[2])[-2:]
    eyes = sorted(eyes, key=lambda eye: eye[0])
    return eyes[0], eyes[1]


def main():
    parser = util.HelperParser(
        description='OpenCV faces utilities using cascade detector.')
    parser.add_argument('-f', '--face_classifier', required=True,
                        help='The face classifier to be used.')
    parser.add_argument('-e', '--eye_classifier', required=True,
                        help='The eye classifier to be used.')
    parser.add_argument(
        '-u', '--url', help='The stream name (can be an url or the device name).')

    args = parser.parse_args()

    cap = util.BufferlessVideoCapture(args.url)
    if not cap.isOpened():
        print("Cannot open stream")
        exit()
    else:
        print("Stream opened")

    face_detector = cv.CascadeClassifier(args.face_classifier)
    eye_detector = cv.CascadeClassifier(args.eye_classifier)

    killer = util.GracefulKiller()

    while not killer.kill_now:
        frame = cap.read()
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = detect_faces(frame, face_detector, eye_detector,
                             max_faces=1,
                             rotations=[-45.0, -22.5, 0.0, 22.5, 45.0])
        face_found = None if len(faces) == 0 else faces[0]

        if face_found is not None:
            mat = face_found.mat
            w = len(mat[0])
            h = len(mat)
            frame[0:h, 0: w] = mat
            (x, y, w, h) = face_found.bbox
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv.imshow('frame', frame)
        cv.waitKey(1)
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
