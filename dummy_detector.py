import threading
import time
import util
import detect_it_all_bot


class DummyDetector():
    def __init__(self):
        self.running = True
        self.detections = {}
        self.callback = self.log_callback
        t = threading.Thread(target=self._send_detections)
        t.daemon = True
        t.start()

    def log_callback(self, chat_id, detection_text):
        print(f"{chat_id}: {detection_text}")

    def stop(self):
        self.running = False

    def _send_detections(self):
        while self.running:
            detections = self.detections.copy()
            self.detections.clear()
            for user_id, user_detections in detections.items():
                user_detections = detections[user_id]
                for d in user_detections:
                    detection_text = f"I detected {d}."
                    self.callback(user_id, detection_text)
            time.sleep(0.5)

    def detect(self, user_id, args):
        if len(args) > 0:
            whatever = args[0]
            user_detections = self.detections[user_id] if user_id in self.detections else set()
            user_detections.add(whatever)
            self.detections[user_id] = user_detections
            return f"Detection of {whatever} in progress."
        else:
            return f"You have to give me something to detect."

    def describe(self):
        return """
This is just a dummy detector.
It detects whatever your want once.
Just send '/detect <whatever-you-want>'.
        """


def main():
    parser = util.HelperParser(
        description='Just a dummy detector.')
    parser.add_argument('-k', '--token', required=True,
                        help='The telegram bot token.')
    parser.add_argument('-p', '--password', required=True,
                        help='The telegram bot client password.')

    args = parser.parse_args()
    detector = DummyDetector()
    bot = detect_it_all_bot.DetectItAllBot(args.token, args.password, detector)
    detector.callback = bot.detection_callback

    def stop():
        detector.stop()
        bot.stop()

    killer = util.GracefulKiller()
    killer.exit_func = stop
    bot.start()


if __name__ == "__main__":
    main()
