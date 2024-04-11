import torch
import numpy as np


class TypeHelper:
    """
    Gets data type information of the ONNX Runtime inference session and provides the mapping from
    `OrtValue` data types to the data types of other frameworks (NumPy, PyTorch, etc).
    """

    @staticmethod
    def ort_type_to_numpy_type(ort_type: str):
        ort_type_to_numpy_type_map = {
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(int8)": np.int8,
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(bool)": bool,
        }
        if ort_type in ort_type_to_numpy_type_map:
            return ort_type_to_numpy_type_map[ort_type]
        else:
            raise ValueError(
                f"{ort_type} is not supported.Here is a list of supported data type:{ort_type_to_numpy_type_map.keys()}"
            )

    @staticmethod
    def ort_type_to_torch_type(ort_type: str):
        ort_type_to_torch_type_map = {
            "tensor(int64)": torch.int64,
            "tensor(int32)": torch.int32,
            "tensor(int8)": torch.int8,
            "tensor(float)": torch.float32,
            "tensor(float16)": torch.float16,
            "tensor(bool)": torch.bool,
        }
        if ort_type in ort_type_to_torch_type_map:
            return ort_type_to_torch_type_map[ort_type]
        else:
            raise ValueError(
                f"{ort_type} is not supported.Here is a list of supported data type:{ort_type_to_torch_type_map.keys()}"
            )


class IOBindingHelper:
    @staticmethod
    def get_device_index(device):
        if isinstance(device, str):
            # could be 'cuda:0', 'cuda:1', or 'cpu'. with cpu, set index=0
            device = torch.device(device)
        elif isinstance(device, int):
            return device
        return 0 if device.index is None else device.index
