import json
import os

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       "error_messages.json")) as f:
    error_messages = json.load(f)


class BadRequestException(Exception):
    def __init__(self, code, field, *args):
        self.code = code
        self.field = field
        error_message = error_messages[code]
        if args is not None:
            self.message = error_message.format(*args)


class VmsSdkException(Exception):
    def __init__(self, message):
        self.message = message


class InternalException(Exception):
    def __init__(self, message):
        self.message = message
