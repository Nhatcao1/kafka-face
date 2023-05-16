import datetime
from readline import get_endidx

import cv2
import numpy as np
import os
import subprocess as sp
import shlex
from service.storage_service import upload_video
from core.config import get_app_settings


settings = get_app_settings()


class VideoWriter:
    def __init__(self, video_duration: int, fps: int, config_id: str, video_type: str, feature_name: str):
        self.frames = []
        self.fps = fps
        self.max_frame = video_duration * fps
        self.save_video = False
        self.video_type = video_type
        self.video_path = ""
        self.object_name = ""
        self.config_id = config_id
        self.feature_name = feature_name

    def add_frame(self, frame: np.ndarray):
        self.frames.append(frame)
        if not self.save_video:
            if len(self.frames) == int(self.max_frame/2.0):
                self.frames.pop(0)
        else:
            if len(self.frames) == self.max_frame:
                self.save_video = False
                self._save_frames_to_video()
                self.frames = []    

    def _save_frames_to_video(self):
        frame_height, frame_width, _ = self.frames[0].shape
        if self.video_type == "event":
            output_width, output_height = 1280, 720
            process = sp.Popen(shlex.split(
                f'ffmpeg -y -s {output_width}x{output_height} -pixel_format bgr24 -f rawvideo -r {self.fps}'
                f' -i pipe: -vcodec libx264 -pix_fmt yuv420p -crf 24 {self.video_path}'),
                stdin=sp.PIPE)

        elif self.video_type == "raw":
            output_width, output_height = frame_width, frame_height
            process = sp.Popen(shlex.split(
                f'ffmpeg -y -s {output_width}x{output_height} -pixel_format bgr24 -f rawvideo -r {self.fps}'
                f' -i pipe: -pix_fmt yuv420p -crf 24 {self.video_path}'),
                stdin=sp.PIPE)
        for frame in self.frames:
            # Build synthetic image for testing ("render" a video frame).
            if frame is not None:
                if frame_width != output_width or frame_height != output_height:
                    frame = cv2.resize(frame, (output_width, output_height), cv2.INTER_AREA)
                try:
                    # Write raw video frame to input stream of ffmpeg sub-process.
                    process.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    pass

        # Close and flush stdin
        process.stdin.close()

        # Wait for sub-process to finish
        process.wait()

        # Terminate the sub-process
        process.terminate()
        upload_video(video_path=self.video_path, object_name=self.object_name)

    def build_video_path(self, timestamp_int: int):
        timestamp = datetime.datetime.fromtimestamp((timestamp_int)/1000)
        if self.video_type == "event":
            video_dir = "videos"
        else:
            video_dir = "raw_videos"
        dir_path = "{}/{}/{}/{}/{}".format(video_dir, timestamp.year, timestamp.month, timestamp.day, self.config_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return "{}/{}_{}_{}.mp4".format(
            dir_path,
            timestamp.strftime("%H"),
            timestamp.strftime("%M"),
            timestamp.strftime("%S")
        )

    def get_video_url(self, timestamp):
        if not self.save_video:
            self.video_path = self.build_video_path(timestamp)
            video_path = self.video_path.replace("videos/", "")
            self.object_name = f"{self.feature_name}/{video_path}"
        return f"iva-video/{self.object_name}"
