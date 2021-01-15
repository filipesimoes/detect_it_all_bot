import cv2 as cv
import util
import threading
import detect_it_all_bot
import time

class CascadeDetector():

    def __init__(self,
                 classifier_path: str,
                 cap: util.BufferlessVideoCapture,
                 color: int = 0,
                 cooldown: float = 30,
                 visible: bool = False,):
        self.callback = self.log_callback
        self.running = True
        self.classifier = cv.CascadeClassifier(classifier_path)
        self.cap = cap
        self.color = color
        self.cooldown = cooldown
        self.visible = visible
        self.users = set()
        self.user_cooldown = {}
        t = threading.Thread(target=self._run_detection)
        t.daemon = True
        t.start()

    def log_callback(self, chat_id, detection_text, frame=None):
        print(f"{chat_id}: {detection_text}")

    def describe(self):
        return f"""
This is a OpenCV cascade detector.
It uses the supplied cascade detector
to detect objects.
Just send '/detect' and we are ready.
        """

    def detect(self, user_id, args):
        self.users.add(user_id)
        self.user_cooldown[user_id] = 0
        return f"Detection in progress."

    def _run_detection(self):
        while self.running:
            if len(self.users) > 0 or self.visible:
                frame = self.cap.read()
                objects = self.detect_in_frame(frame)
                if len(objects) > 0:
                    detection_text = f"{len(objects)} objects detected."
                    for (x, y, w, h) in objects:
                        cv.rectangle(
                            frame, (x, y), (x + w, y + h), self.color, 2)
                    for user_id in self.users:
                        if self.user_cooldown[user_id] < time.time():
                            self.user_cooldown[user_id] = time.time(
                            ) + self.cooldown
                            self.callback(user_id, detection_text, frame)
                if self.visible:
                    cv.imshow('cascade_detector', frame)
                    cv.waitKey(1)

    def stop(self):
        self.running = False

    def detect_in_frame(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        return self.classifier.detectMultiScale(gray, 1.2, 5)


def main():
    parser = util.HelperParser(
        description='OpenCV cascade detector.')
    parser.add_argument('-k', '--token', required=True,
                        help='The telegram bot token.')
    parser.add_argument('-p', '--password', required=True,
                        help='The telegram bot client password.')
    parser.add_argument('-c', '--classifier', required=True,
                        help='The classifier to be used.')
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
    detector = CascadeDetector(
        args.classifier,
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
