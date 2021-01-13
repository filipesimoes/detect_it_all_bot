from telegram.ext import (
    Updater,
    CommandHandler,
    Filters,
    MessageHandler,
    CallbackContext,
)
import util
import signal
import logging
import dummy_detector
import cv2 as cv
import io

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class DetectItAllBot:
    def __init__(self, token, auth_token, detector):
        self.updater = Updater(token=token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        self.auth_token = auth_token
        self.init_handlers()
        self.authenticated_clients = set()
        self.detector = detector

    def init_handlers(self):
        start_handler = CommandHandler('start', self.start_cmd)
        detect_handler = CommandHandler('detect', self.detect_cmd)
        describe_handler = CommandHandler('describe', self.describe_cmd)
        echo_handler = MessageHandler(
            Filters.text & (~Filters.command), self.echo_cmd)
        self.dispatcher.add_handler(start_handler)
        self.dispatcher.add_handler(detect_handler)
        self.dispatcher.add_handler(describe_handler)
        self.dispatcher.add_handler(echo_handler)

    def start(self):
        print("Starting bot...")
        self.updater.start_polling()
        print("Started!")

    def stop(self):
        print("Stopping bot...")
        self.updater.stop()

    def is_authenticated(self, update):
        return update.effective_chat.id in self.authenticated_clients

    def send_unauthenticated_error(self, update, context):
        message = """
You are not authenticated!
To authenticate send a command '/start <your-secret-code>'.
            """
        context.bot.send_message(
            chat_id=update.effective_chat.id, text=message)

    def start_cmd(self, update, context: CallbackContext):
        client_token_correct = len(
            context.args) > 0 and context.args[0] == self.auth_token
        if (client_token_correct):
            self.authenticated_clients.add(update.effective_chat.id)
            message = """
You are authenticated!
You can now use the other commands.
            """
            context.bot.send_message(
                chat_id=update.effective_chat.id, text=message)
        else:
            self.send_unauthenticated_error(update, context)

    def echo_cmd(self, update, context: CallbackContext):
        context.bot.send_message(
            chat_id=update.effective_chat.id, text=update.message.text)

    def detect_cmd(self, update, context: CallbackContext):
        if self.is_authenticated(update):
            chat_id = chat_id = update.effective_chat.id
            detection_msg = self.detector.detect(chat_id, context.args)
            context.bot.send_message(chat_id=chat_id, text=detection_msg)
        else:
            self.send_unauthenticated_error(update, context)

    def detection_callback(self, chat_id, detection_text, frame=None):
        def send_detection(context: CallbackContext):
            context.bot.send_message(chat_id=chat_id, text=detection_text)
            if frame is not None:
                photo = io.BytesIO(cv.imencode('.jpg', frame)[1])
                context.bot.send_photo(chat_id=chat_id, photo=photo)
        self.updater.job_queue.run_once(send_detection, 0)

    def describe_cmd(self, update, context: CallbackContext):
        if self.is_authenticated(update):
            chat_id = chat_id = update.effective_chat.id
            description = self.detector.describe()
            context.bot.send_message(chat_id=chat_id, text=description)
        else:
            self.send_unauthenticated_error(update, context)
