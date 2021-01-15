import cv2 as cv
import util
import threading
import detect_it_all_bot
import time


class LBPHFacedDetector():

    def __init__(self,
                 classifier_path: str,
                 recognizer_path: str,
                 names,
                 cap: util.BufferlessVideoCapture,
                 color: int = 0,
                 cooldown: float = 30,
                 min_probability: float = 50,
                 visible: bool = False,):
        self.running = True
        self.callback = self.log_callback
        self.classifier = cv.CascadeClassifier(classifier_path)
        self.recognizer = cv.face.LBPHFaceRecognizer_create()
        self.recognizer.read(recognizer_path)
        self.names = names
        self.cap = cap
        self.color = color
        self.cooldown = cooldown
        self.min_probability = min_probability
        self.visible = visible
        self.users = set()
        self.user_cooldown = {}
        t = threading.Thread(target=self._run_detection)
        t.daemon = True
        t.start()

    def log_callback(self, chat_id, detection_text):
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
                    for face in faces:
                        (x, y, w, h) = face["coords"]
                        face_id = face["face_id"]
                        name = face["name"]
                        probability = face["probability"]
                        text = f"{name} ({face_id}): {probability:.1f}"
                        cv.rectangle(
                            frame, (x, y), (x + w, y + h), self.color, 2)
                        cv.putText(frame, text, (x, y - 5),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 1)
                    for user_id in self.users:
                        if self.user_cooldown[user_id] < time.time():
                            self.user_cooldown[user_id] = time.time(
                            ) + self.cooldown
                            self.callback(user_id, detection_text, frame)
                if self.visible:
                    cv.imshow('lbph_face_detector', frame)
                    cv.waitKey(1)
            pass

    def detect_in_frame(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = self.classifier.detectMultiScale(gray, 1.2, 5)
        recognized_faces = []
        for face_coords in faces:
            (x, y, w, h) = face_coords
            face = gray[y:y+h, x:x+w]
            face_id, confidence = self.recognizer.predict(face)
            probability = 100 - confidence
            if probability >= self.min_probability:
                recognized_faces.append({
                    "face_id": face_id,
                    "probability": probability,
                    "name": self.names[face_id],
                    "coords": face_coords,
                })
        return recognized_faces

    def describe(self):
        return f"""
This is a OpenCV cascade classifier and
LBPH face recognizer to detect
and recognize faces.
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
    parser.add_argument('-r', '--recognizer', required=True,
                        help='The recognizer to be used.')
    parser.add_argument('-n', '--names', nargs="+", required=True,
                        help='The names list to be used.')
    parser.add_argument('-d', '--cooldown', default=30, type=float,
                        help='The cooldown after a detection in seconds. Default: 30')
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
    detector = LBPHFacedDetector(
        args.classifier,
        args.recognizer,
        args.names,
        cap,
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
