import os
import sys
import traceback

import torch


def print_value(name, value):
    print(f"{name}={value}")


def main():
    print("=== RVC CUDA Debug ===")
    print_value("python", sys.version.replace("\n", " "))
    print_value("torch", torch.__version__)
    print_value("torch.version.cuda", getattr(torch.version, "cuda", None))

    for env_name in (
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "HF_HOME",
    ):
        print_value(env_name, os.getenv(env_name, ""))

    for device_path in ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia0"):
        print_value(f"{device_path}.exists", os.path.exists(device_path))

    try:
        cuda_available = torch.cuda.is_available()
        print_value("torch.cuda.is_available()", cuda_available)
    except Exception as exc:
        print_value("torch.cuda.is_available()_error", repr(exc))
        cuda_available = False

    try:
        device_count = torch.cuda.device_count()
        print_value("torch.cuda.device_count()", device_count)
    except Exception as exc:
        print_value("torch.cuda.device_count()_error", repr(exc))
        device_count = 0

    if not cuda_available and device_count > 0:
        print(
            "NOTE: device_count > 0 while cuda.is_available() is False. "
            "This usually means the driver is partially visible but CUDA initialization is failing."
        )

    for index in range(device_count):
        try:
            props = torch.cuda.get_device_properties(index)
            print_value(f"device[{index}].name", props.name)
            print_value(
                f"device[{index}].total_memory_gb",
                round(props.total_memory / 1024 / 1024 / 1024, 2),
            )
        except Exception as exc:
            print_value(f"device[{index}].probe_error", repr(exc))
            traceback.print_exc()

    print("=== End RVC CUDA Debug ===")


if __name__ == "__main__":
    main()
