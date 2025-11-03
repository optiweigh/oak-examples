import depthai as dai
from depthai_nodes.node import GatherData
from depthai_nodes.node.utils import generate_script_content
from .mosaic_layout_node import MosaicLayoutNode


def build_roi_cropper(
    pipeline: dai.Pipeline,
    *,
    preview_stream: dai.Node.Output,
    det_stream: dai.Node.Output,
    out_size: tuple[int, int],
    frame_type: dai.ImgFrame.Type,
    pool_size: int = 50,
    cfg_queue_size: int = 50,
) -> dai.Node.Output:
    """
    Build: Script(det_in+preview) -> ImageManip(crop+resize to out_size, frame_type) -> out
    Returns cropper.out (frames of size out_size)
    """
    resize_w, resize_h = out_size
    script = pipeline.create(dai.node.Script)
    script.setScript(generate_script_content(resize_width=resize_w, resize_height=resize_h))
    det_stream.link(script.inputs["det_in"])
    preview_stream.link(script.inputs["preview"])

    manip = pipeline.create(dai.node.ImageManip)
    manip.setMaxOutputFrameSize(resize_w * resize_h * 3)
    manip.initialConfig.setOutputSize(resize_w, resize_h)
    manip.initialConfig.setFrameType(frame_type)
    manip.inputConfig.setMaxSize(cfg_queue_size)
    manip.inputImage.setMaxSize(cfg_queue_size)
    manip.setNumFramesPool(pool_size)
    manip.inputConfig.setWaitForMessage(True)

    script.outputs["manip_cfg"].link(manip.inputConfig)
    script.outputs["manip_img"].link(manip.inputImage)

    return manip.out


def convert_to_nv12(pipeline: dai.Pipeline, original: dai.Node.Output, width: int, height: int) -> dai.Node:
    """
    ImageManip to convert to NV12 at given size, returns the ImageManip node (use .out)
    """
    im = pipeline.create(dai.node.ImageManip)
    im.initialConfig.setOutputSize(width, height)
    im.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)
    im.setMaxOutputFrameSize(int(width * height * 3))
    original.link(im.inputImage)
    return im


def build_h264_stream(
    pipeline: dai.Pipeline,
    *,
    src: dai.Node.Output,
    size: tuple[int, int],
    fps: int,
    profile: dai.VideoEncoderProperties.Profile,
    assume_nv12: bool = False,
    vbr: bool = False,
) -> dai.Node.Output:
    """
    Generic encoder builder
    - If assume_nv12=True: links src directly to encoder (expects NV12 at 'size')
    - Else: inserts NV12 converter+resize to 'size' before encoder
    Returns encoder.out (bitstream)
    """
    if assume_nv12:
        enc_in = src
    else:
        nv12_node = convert_to_nv12(pipeline, src, size[0], size[1])
        enc_in = nv12_node.out

    enc = pipeline.create(dai.node.VideoEncoder)
    enc.setDefaultProfilePreset(fps=fps, profile=profile)
    if vbr:
        enc.setRateControlMode(dai.VideoEncoderProperties.RateControlMode.VBR)
    enc_in.link(enc.input)
    return enc.out


def build_mosaic_from_crops(
    pipeline: dai.Pipeline,
    *,
    fps_limit: int,
    crops_stream: dai.Node.Output,
    reference_stream: dai.Node.Output,
    target_size: tuple[int, int],
    frame_type: dai.ImgFrame.Type,
) -> dai.Node.Output:
    """
    One-shot: GatherData(crops + reference) -> MosaicLayoutNode(target_size)
    Returns mosaic.output
    """
    gather = pipeline.create(GatherData).build(fps_limit)
    crops_stream.link(gather.input_data)
    reference_stream.link(gather.input_reference)

    mosaic = pipeline.create(MosaicLayoutNode).build(
        crops_input=gather.out,
        target_size=target_size,
        frame_type=frame_type,
    )
    mosaic.frame_type = frame_type
    return mosaic.output


def build_resizer(
    pipeline: dai.Pipeline,
    *,
    input_stream: dai.Node.Output,
    size: tuple[int, int],
    frame_type: dai.ImgFrame.Type,
    mode: dai.ImageManipConfig.ResizeMode = dai.ImageManipConfig.ResizeMode.LETTERBOX,
    max_output_bytes: int | None = None,
) -> dai.node.ImageManip:
    """
    Create an ImageManip that resizes (and optionally letterboxes/stretches) frames
    to `size`, outputs as `frame_type`, and links `input_stream` -> manip.inputImage

    Returns the ImageManip node (use `.out` to consume)
    """
    w, h = size

    # heuristic for buffer size if not provided
    if max_output_bytes is None:
        # NV12 ≈ 1.5 bytes/pixel, most others here ≈ 3 bytes/pixel
        bpp = 1.5 if frame_type == dai.ImgFrame.Type.NV12 else 3.0
        max_output_bytes = int(w * h * bpp)

    manip = pipeline.create(dai.node.ImageManip)
    manip.setMaxOutputFrameSize(max_output_bytes)
    manip.initialConfig.setOutputSize(w, h, mode=mode)
    manip.initialConfig.setFrameType(frame_type)

    input_stream.link(manip.inputImage)
    return manip


def load_model(model: str, platform: str) -> tuple[int, int, dai.NNArchive]:
    """
    Load a model from the DepthAI model zoo, set its platform, and return its input size and archive

    Returns (input_width, input_height, model_archive)
    """
    model_description = dai.NNModelDescription(model)
    model_description.platform = platform
    model_archive = dai.NNArchive(dai.getModelFromZoo(model_description))
    return model_archive.getInputWidth(), model_archive.getInputHeight(), model_archive
