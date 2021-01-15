import numpy as np
import cv2 as cv
import util
import threading
import time
from os.path import abspath
import detect_it_all_bot
import logging


class YOLODetector():
    def __init__(self,
                 configuration,
                 weights,
                 classes,
                 cap: util.BufferlessVideoCapture,
                 confidence_threshold: float = 0.5,
                 cooldown: float = 30,
                 visible: bool = False,
                 selected_classes=[]):
        self.running = True
        self.callback = self.log_callback
        self.net = cv.dnn.readNetFromDarknet(configuration, weights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1]
                   for i in self.net.getUnconnectedOutLayers()]
        self.classes = classes
        self.cap = cap
        self.confidence_threshold = confidence_threshold
        self.cooldown = cooldown
        self.visible = visible
        self.colors = np.random.randint(
            0, 255, size=(len(self.classes), 3), dtype='uint8')
        self.selected_classes = selected_classes
        self.user_detections = {}
        self.user_cooldown = {}
        t = threading.Thread(target=self._run_detection)
        t.daemon = True
        t.start()

    def log_callback(self, chat_id, detection_text, frame=None):
        print(f"{chat_id}: {detection_text}")

    def describe(self):
        available_classes = self.selected_classes.join('\n')
        return f"""
This is a YOLOv3 detector.
It detects the classes you select.
Just send '/detect <class-1> <class-2> ... <class-n>'.
The available classes are:
{available_classes}
        """

    def detect(self, user_id, args):
        if len(args) > 0:
            user_detections = self.user_detections[user_id] if user_id in self.user_detections else set(
            )
            [user_detections.add(c) for c in args]
            self.user_detections[user_id] = user_detections
            self.user_cooldown[user_id] = 0
            selection = ', '.join(user_detections)
            return f"Detection of {selection} in progress."
        else:
            return "You have to give me something to detect."

    def _run_detection(self):
        while self.running:
            if len(self.user_detections) > 0 or self.visible:
                t0 = time.time()
                frame = self.cap.read()
                results = self.detect_in_frame(frame)
                logging.debug(f'Detected {len(results)} objects.')
                for result in results:
                    (x, y) = result["pos"]
                    (w, h) = result["size"]
                    color = result["color"]
                    classID = result["classID"]
                    confidence = result["confidence"]
                    text = f"{classID}: {confidence:.2f}"
                    cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv.putText(frame, text, (x, y - 5),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                t = time.time()
                detection_text = None
                user_detections = self.user_detections.copy()
                for user_id, detections in user_detections.items():
                    if any([r["classID"] in detections for r in results]) and self.user_cooldown[user_id] < time.time():
                        self.user_cooldown[user_id] = time.time(
                        ) + self.cooldown
                        if detection_text is None:
                            detection_text = f"Detection of:\n{self.create_detection_text(results)}."
                        self.callback(user_id, detection_text, frame)

                cv.putText(frame, f"{1/(t-t0):.1f}fps", (5, 15),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
                if self.visible:
                    cv.imshow('yolo_detector', frame)
                    cv.waitKey(1)
            time.sleep(0.5)

    def create_detection_text(self, results):
        results_text = []
        for r in results:
            classID = r["classID"]
            confidence = r["confidence"]
            results_text.append(f"{classID}: {confidence:.4f}")
        return "\n".join(results_text)

    def stop(self):
        self.running = False

    def detect_in_frame(self, frame):
        blob = cv.dnn.blobFromImage(
            frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.ln)
        boxes = []
        confidences = []
        classIDs = []
        h, w = frame.shape[:2]
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > self.confidence_threshold and (len(self.selected_classes) == 0 or classID in self.selected_classes):
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        indices = cv.dnn.NMSBoxes(
            boxes, confidences, self.confidence_threshold, 0.4)
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append(
                    {
                        "pos": (boxes[i][0], boxes[i][1]),
                        "size": (boxes[i][2], boxes[i][3]),
                        "color": [int(c) for c in self.colors[classIDs[i]]],
                        "confidence": confidences[i],
                        "classID": self.classes[classIDs[i]],
                    }
                )
        return results


def main():
    parser = util.HelperParser(
        description='YOLO Object Detection/Classification bot.')
    parser.add_argument('-l', '--labels', default='coco.names',
                        help='Label names. Default: coco.names.')
    parser.add_argument('-c', '--configuration', default='yolov3.cfg',
                        help='YOLO configuration file. Default: yolov3.cfg.')
    parser.add_argument('-w', '--weights', default='yolov3.weights',
                        help='YOLO weights file. Default: yolov3.weights.')
    parser.add_argument('-t', '--confidence_threshold', type=float,
                        default=0.5, help='The confidence threshold. Default: 0.5.')
    parser.add_argument(
        '-u', '--url', help='The stream name (can be an url or the device name).')
    parser.add_argument('-k', '--token', help='The telegram bot token.')
    parser.add_argument('-p', '--password', required=True,
                        help='The telegram bot client password.')
    parser.add_argument('-d', '--cooldown', default=30, type=float,
                        help='The cooldown after a detection in seconds. Default: 30')
    parser.add_argument('-v', '--visible', default=False, type=bool,
                        help='Show detect window. Default: False')

    args = parser.parse_args()

    cap = util.BufferlessVideoCapture(args.url)
    if not cap.isOpened():
        print("Cannot open stream")
        exit()
    else:
        print("Stream opened")

    classes = open(abspath(args.labels)).read().strip().split('\n')
    configuration = abspath(args.configuration)
    weights = abspath(args.weights)

    detector = YOLODetector(configuration, weights,
                            classes, cap, args.confidence_threshold, args.cooldown, args.visible)
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
