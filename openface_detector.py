import cv2 as cv
import util
import threading
import detect_it_all_bot
import time
from faces_util import detect_faces
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import openface_util
import pandas as pd
import numpy as np


class OpenfaceDetector():

    def __init__(self,
                 classifier_path: str,
                 eye_classifier_path: str,
                 model_path: str,
                 labels_path,
                 reps_path,
                 cap: util.BufferlessVideoCapture,
                 color=(255, 0, 0),
                 minimum_probability: float = 0.5,
                 cooldown: float = 30,
                 img_dim: int = 96,
                 visible: bool = False,
                 cuda: bool = False,
                 openface_dir: str = "/usr/local/lib/python3.9/dist-packages/openface"):
        self.running = True
        self.callback = self.log_callback
        self.face_classifier = cv.CascadeClassifier(classifier_path)
        self.eye_classifier = cv.CascadeClassifier(eye_classifier_path)
        self.net = openface_util.TorchNeuralNet(
            openface_dir, model_path, imgDim=img_dim, cuda=cuda)
        self.labels_path = labels_path
        self.reps_path = reps_path
        self.cap = cap
        self.color = color
        self.minimum_probability = minimum_probability
        self.cooldown = cooldown
        self.visible = visible
        self.users = set()
        self.user_cooldown = {}
        self.train()
        t = threading.Thread(target=self._run_detection)
        t.daemon = True
        t.start()

    def train(self):
        labels = pd.read_csv(self.labels_path, header=None)[0].to_list()
        self.le = LabelEncoder().fit(labels)
        num_classes = len(self.le.classes_)
        print(f"Training for {num_classes} classes.")
        t0 = time.time()
        reps = pd.read_csv(self.reps_path, header=None).values
        labels_num = self.le.transform(labels)
        self.classifier = SVC(C=1, kernel='linear', probability=True)
        self.classifier.fit(reps, labels_num)
        took = time.time() - t0
        print(f"Training took {took}")

    def log_callback(self, chat_id, detection_text, frame):
        print(f"{chat_id}: {detection_text}")

    def stop(self):
        self.running = False

    def _run_detection(self):
        while self.running:
            if len(self.users) > 0 or self.visible:
                frame = self.cap.read()
                faces = self.detect_in_frame(frame)
                if len(faces) > 0:
                    detection_text = f"{len(faces)} faces detected."
                    detected = False
                    for face in faces:
                        (x, y, w, h) = face["bbox"]
                        face_id = face["face_id"]
                        name = face["name"]
                        confidence = face["confidence"]
                        if self.visible:
                            text = f"{name} ({face_id}): {confidence:.2f}"
                            cv.rectangle(
                                frame, (x, y), (x + w, y + h), self.color, 2)
                            cv.putText(frame, text, (x, y - 5),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 1)
                        detected = False  # detected or probability >= self.minimum_probability
                    if detected:
                        for user_id in self.users:
                            if self.user_cooldown[user_id] < time.time():
                                self.user_cooldown[user_id] = time.time(
                                ) + self.cooldown
                                self.callback(user_id, detection_text, frame)
                if self.visible:
                    cv.imshow('openface', frame)
                    cv.waitKey(1)

    def detect_in_frame(self, frame):
        recognized_faces = []
        faces = detect_faces(frame,
                             self.face_classifier,
                             self.eye_classifier,
                             desired_face_width=96,
                             desired_face_height=96,)
        for face in faces:
            rep = self.net.forward(face.mat)
            #rep = rep[1].reshape(1, -1)
            predictions = self.classifier.predict_proba(rep).ravel()
            max_i = np.argmax(predictions)
            name = self.le.inverse_transform(max_i)
            confidence = predictions[max_i]
            recognized_faces.append({
                "bbox": face.bbox,
                "name": name,
                "face_id": max_i,
                "confidence": confidence,
            })

        return recognized_faces

    def describe(self):
        return f"""
This is a OpenCV cascade classifier
to detect faces and Openface to
recognize faces.
Just send '/detect' and we are ready.
        """

    def detect(self, user_id, args):
        self.users.add(user_id)
        self.user_cooldown[user_id] = 0
        return f"Detection in progress."


def main():
    parser = util.HelperParser(
        description='OpenCV cascade classifier and LBPH face recognizer.')
    parser.add_argument('-k', '--token', required=True,
                        help='The telegram bot token.')
    parser.add_argument('-p', '--password', required=True,
                        help='The telegram bot client password.')
    parser.add_argument('-c', '--classifier', required=True,
                        help='The classifier to be used.')
    parser.add_argument('-e', '--eye_classifier', required=True,
                        help='The eye classifier to be used.')
    parser.add_argument('-t', '--torch_face_model', required=True,
                        help='The torch network model to be used.')
    parser.add_argument('-l', '--labels', required=True,
                        help='The labels path.')
    parser.add_argument('-s', '--representations', required=True,
                        help='The representations path.')
    parser.add_argument('-d', '--cooldown', default=30, type=float,
                        help='The cooldown after a detection in seconds. Default: 30')
    parser.add_argument('-m', '--minimum_probability', default=50, type=float,
                        help='The minimum probability to accept a face as a match. Default: 50')
    parser.add_argument(
        '-u', '--url', help='The stream name (can be an url or the device name).')
    parser.add_argument('-v', '--visible', default=False, type=bool,
                        help='Show detect window. Default: False')

    args = parser.parse_args()

    cap = util.BufferlessVideoCapture(args.url)
    if not cap.isOpened():
        print("Cannot open stream")
        exit()
    else:
        print("Stream opened")

    args = parser.parse_args()
    detector = OpenfaceDetector(
        args.classifier,
        args.eye_classifier,
        args.torch_face_model,
        args.labels,
        args.representations,
        cap,
        minimum_probability=args.minimum_probability,
        cooldown=args.cooldown,
        visible=args.visible,
    )
    bot = detect_it_all_bot.DetectItAllBot(args.token, args.password, detector)
    detector.callback = bot.detection_callback

    def stop():
        cap.release()
        detector.stop()
        bot.stop()

    killer = util.GracefulKiller()
    killer.exit_func = stop
    bot.start()


if __name__ == "__main__":
    main()
