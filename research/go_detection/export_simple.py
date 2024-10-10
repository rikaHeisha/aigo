import os

import debugpy
import torch
import torch.nn as nn
from go_detection.export_util import export_model


class SampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor([2.0]), requires_grad=True)

    def forward(self, x):
        return (self.weights * x).sum(dim=0)


# def main(exp_name: str, export_path: str = "model.tflite"):
def main():
    model = SampleModel()
    input_tensor = torch.rand(3)

    print("\033[32mAbout to export sample model\033[0m")
    export_model(
        model,
        "/home/rmenon/Desktop/dev/projects/aigo/research/model.tflite",
        input_tensor.shape,
    )


if __name__ == "__main__":
    if os.environ.get("ENABLE_DEBUGPY"):
        print("")
        print("\033[31mWaiting for debugger to connect\033[0m")
        debugpy.listen(5678)
        debugpy.wait_for_client()
    main()
