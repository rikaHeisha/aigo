import logging
import os
from typing import Tuple

import ai_edge_torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def export_model(model, export_path: str, input_shape: Tuple):
    original_env = os.environ.copy()
    os.environ["PJRT_DEVICE"] = "CPU"
    model = model.cpu().eval()

    input_tensor = torch.rand(*input_shape)
    expected_output = model(input_tensor)

    edge_model = ai_edge_torch.convert(model, (input_tensor,))
    model_output = edge_model(input_tensor)

    if np.isclose(expected_output.detach().numpy(), model_output, atol=1e-5).all():
        # Success
        logger.info(f"Exporting model to: {export_path}")
        edge_model.export(export_path)
    else:
        # Failed
        logger.warning(
            "Failed to export model. Expected output does not match actual output\n"
            f"Expected output: {expected_output}\n"
            f"Model output: {model_output}"
        )

    os.environ = original_env
