# config/system_configuration.py
from __future__ import annotations

from .config_data_classes import VideoConfig, ModelNames

from typing import Optional
from argparse import Namespace
import depthai as dai


class SystemConfiguration:
    """
    Class that manages configuration initialization.
    """

    def __init__(self, platform: str, args: Namespace):
        self._platform = platform
        self._args = args
        self._video_config: Optional[VideoConfig] = None
        self._model_names: Optional[ModelNames] = None

    def build(self) -> "SystemConfiguration":
        if self._args.fps_limit is None:
            self._args.fps_limit = 15
            print(
                f"\nFPS limit set to {self._args.fps_limit} for platform {self._platform}. "
                f"If you want to set a custom FPS limit, use the --fps_limit flag.\n"
            )

        self._video_config = VideoConfig(
            fps=self._args.fps_limit,
            frame_type=dai.ImgFrame.Type.BGR888i,
            width=1280,
            height=720,
        )

        self._model_names = ModelNames(
            face="luxonis/yunet:640x360",
            emotions="luxonis/emotion-recognition:260x260",
            age_gender="luxonis/age-gender-recognition:62x62",
            people="luxonis/yolov6-nano:r2-coco-512x288",
            reid="luxonis/arcface:lfw-112x112",
        )

        return self

    def get_video_config(self) -> VideoConfig:
        if self._video_config is None:
            raise RuntimeError(
                "SystemConfiguration.build() must be called before get_video_config()."
            )
        return self._video_config

    def get_model_names(self) -> ModelNames:
        if self._model_names is None:
            raise RuntimeError(
                "SystemConfiguration.build() must be called before get_model_names()."
            )
        return self._model_names

    @property
    def args(self) -> Namespace:
        return self._args
