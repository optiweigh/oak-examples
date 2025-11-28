import depthai as dai
from typing import Optional
from config.config_data_classes import VideoConfig


class CameraSource:
    """
    Builds the camera part of the pipeline:

    - Creates a Camera node.
    - Provides a BGR888i preview stream.
    - Provides an H.264 encoded stream.
    """

    def __init__(
        self,
        pipeline: dai.Pipeline,
        cfg: VideoConfig,
    ):
        self._pipeline = pipeline
        self._config: VideoConfig = cfg

        self._camera: Optional[dai.node.Camera] = None
        self._preview_out: Optional[dai.Node.Output] = None
        self._nv12_out: Optional[dai.Node.Output] = None
        self._encoded_out: Optional[dai.Node.Output] = None

    def build(self) -> "CameraSource":
        self._camera = self._create_camera()
        self._preview_out, self._nv12_out = self._create_camera_outputs()
        self._encoded_out = self._create_encoder_output()
        return self

    def _create_camera(self) -> dai.node.Camera:
        cam = self._pipeline.create(dai.node.Camera).build()
        return cam

    def _create_camera_outputs(self) -> tuple[dai.Node.Output, dai.Node.Output]:
        """Create BGR and NV12 outputs from the camera."""
        camera_out = self._camera.requestOutput(
            size=(self._config.width, self._config.height),
            type=self._config.frame_type,
            fps=self._config.fps,
        )

        nv12_out = self._camera.requestOutput(
            size=(self._config.width, self._config.height),
            type=dai.ImgFrame.Type.NV12,
            fps=self._config.fps,
        )

        return camera_out, nv12_out

    def _create_encoder_output(self) -> dai.Node.Output:
        encoder = self._pipeline.create(dai.node.VideoEncoder)
        encoder.setDefaultProfilePreset(
            fps=self._config.fps,
            profile=dai.VideoEncoderProperties.Profile.H264_MAIN,
        )
        self._nv12_out.link(encoder.input)
        return encoder.out

    @property
    def preview(self) -> dai.Node.Output:
        """BGR preview stream from the camera."""
        if self._preview_out is None:
            raise RuntimeError(
                "CameraSource.build() must be called before accessing preview."
            )
        return self._preview_out

    @property
    def encoded(self) -> dai.Node.Output:
        """H.264 encoded stream."""
        if self._encoded_out is None:
            raise RuntimeError(
                "CameraSource.build() must be called before accessing encoded."
            )
        return self._encoded_out
