import datetime
import time
import cv2
import os


def build_image_path(config_id: str, timestamp_int: int):
    timestamp = datetime.datetime.fromtimestamp((timestamp_int)/1000)
    dir_path = "raw_images/{}/{}/{}/{}".format(timestamp.year, timestamp.month, timestamp.day, config_id)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return "{}/{}_{}_{}.png".format(
        dir_path,
        timestamp.strftime("%H"),
        timestamp.strftime("%M"),
        timestamp.strftime("%S")
    )


class ImageWriter:
    def __init__(self, config_id: str):
        self.config_id = config_id 

    def save_image(self, image):
        cv2.imwrite(build_image_path(self.config_id, int(time.time()*1000)), image)
