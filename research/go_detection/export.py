import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import ai_edge_torch

import numpy as np
import torchvision


class SampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor([2.0]), requires_grad=True)

    def forward(self, x):
        return (self.weights * x).sum(dim=0)


if __name__ == "__main__":
    os.environ["PJRT_DEVICE"] = "CPU"

    model = SampleModel().eval()
    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    expected_output = model(input_tensor)

    edge_model = ai_edge_torch.convert(model.eval(), (input_tensor,))
    model_output = edge_model(input_tensor)
    edge_model.export("model.tflite")

    print("")
    print(f"Expected output: {expected_output}")
    print(f"Model output   : {model_output}")

    print("Exported model: model.tflite")
    if np.isclose(expected_output.detach().numpy(), model_output, atol=1e-5).all():
        print("\x1b[32mSuccess\x1b[0m")
    else:
        print("\x1b[31mFail\x1b[0m")

    print("")
