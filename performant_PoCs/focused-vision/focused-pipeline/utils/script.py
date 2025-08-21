try:
    GROUP_STRIDE = 1000

    while True:
        frame = node.inputs["frame_input"].get()
        num_msg = node.inputs["num_configs_input"].get()

        gid = int(num_msg.getSequenceNum())
        num_configs = len(bytearray(num_msg.getData()))

        frame.setSequenceNum(gid)
        node.outputs["output_frame"].send(frame)

        for _ in range(max(0, num_configs)):
            cfg = node.inputs["config_input"].get()
            node.outputs["output_config"].send(cfg)

except Exception as e:
    node.warn(f"[Script] Exception: {str(e)}")
