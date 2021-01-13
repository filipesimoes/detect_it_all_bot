import cv2 as cv
import argparse
import threading
import queue
import signal
import sys


class HelperParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f'error: {message}')
        self.print_help()
        sys.exit(2)


class BufferlessVideoCapture:
    def __init__(self, name):
        self.running = True
        self.cap = cv.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        self.cap.release()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


class GracefulKiller:
    kill_now = False

    def __init__(self):
        self.exit_func = self.set_kill_now
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def set_kill_now(self):
        self.kill_now = True

    def exit_gracefully(self, signum, frame):
        self.exit_func()