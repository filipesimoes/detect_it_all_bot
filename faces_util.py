import util
import cv2 as cv
import numpy as np


# Modified from https://github.com/jrosebr1/imutils/blob/master/imutils/face_utils/facealigner.py
def align_face(img,
               face_shape,
               eyes_shapes,
               desired_left_eye=(0.30, 0.30),
               desired_face_width=256,
               desired_face_height=None,):

    if desired_face_height is None:
        desired_face_height = desired_face_width

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
    y = eye_center_y - desired_face_height // 2
    face = img[y:y+h, x:x+w]
    return face


def detect_eyes(eyes):
    assert len(eyes) == 2
    if eyes[0][0] < eyes[1][0]:
        left_eye=eyes[0]
        right_eye=eyes[1]
    else:
        left_eye=eyes[1]
        right_eye=eyes[0]
    return left_eye, right_eye


def main():
    parser=util.HelperParser(
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
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        face_found = None
        if len(faces) > 0:
            face_shape = faces[0]
            (x, y, w, h) = face_shape
            face = gray[y:y+h, x:x+w]
            eyes = eye_detector.detectMultiScale(face, 1.3, 5)
            if len(eyes) == 2:
                face_found = align_face(frame, face_shape, eyes)
            else:
                face_found = None

        if face_found is not None:
            cv.imshow('frame', face_found)
        cv.waitKey(1)
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
