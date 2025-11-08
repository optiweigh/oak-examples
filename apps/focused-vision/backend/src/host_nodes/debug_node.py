
import depthai as dai
import time


class DebugNode(dai.node.HostNode):
    """"""

    def build(self, node_out: dai.Node.Output) -> "DebugNode":
        self.link_args(node_out)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, _any: dai.Buffer) -> None:
        print(f"Input message type: {type(_any)}")
        s = 2
