import os
import sys
from collections import Counter
from dotenv import load_dotenv

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()
load_dotenv("sha256.env")

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

from infer.modules.vc import VC, show_info, hash_similarity
from infer.modules.vc.utils import get_index_path_from_model
from infer.modules.uvr5.modules import uvr
from infer.modules.uvr5.vr import AudioPre
from infer.lib.audio import (
    load_audio,
    get_audio_properties,
    get_supported_sample_rate_for_format,
    resample_audio,
    save_audio,
)
from infer.lib.preset_manager import PresetManager
from infer.lib.train.process_ckpt import (
    change_info,
    extract_small_model,
    merge,
    merge_many,
)
from i18n.i18n import I18nAuto
from configs import Config
from sklearn.cluster import MiniBatchKMeans
import torch, platform
import numpy as np
import gradio as gr
import faiss
import pathlib
import json
from datetime import datetime
from time import sleep
from subprocess import Popen
from random import shuffle
import warnings
import traceback
import threading
import shutil
import logging
import re
import uuid


class _DropBelowError(logging.Filter):
    def filter(self, record):
        return record.levelno >= logging.ERROR


def configure_logging():
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    root = logging.getLogger()
    root.setLevel(logging.ERROR)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root.addHandler(handler)
    for handler in root.handlers:
        handler.setLevel(logging.ERROR)
        if handler.formatter is None:
            handler.setFormatter(formatter)

    config_logger = logging.getLogger("configs.config")
    config_logger.handlers.clear()
    config_logger.setLevel(logging.INFO)
    config_logger.propagate = False
    info_handler = logging.StreamHandler()
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    config_logger.addHandler(info_handler)

    noisy_loggers = [
        "asyncio",
        "fairseq",
        "gradio",
        "httpcore",
        "httpx",
        "matplotlib",
        "numba",
        "PIL",
        "python_multipart",
        "urllib3",
        "uvicorn",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.ERROR)

    pil_image_logger = logging.getLogger("PIL.Image")
    pil_image_logger.setLevel(logging.ERROR)
    pil_image_logger.addFilter(_DropBelowError())


configure_logging()

logger = logging.getLogger(__name__)


def print_cuda_debug_report(config):
    print("=== RVC CUDA Debug ===")
    print(f"torch={torch.__version__}")
    print(f"torch.version.cuda={getattr(torch.version, 'cuda', None)}")
    print(f"config.device={getattr(config, 'device', None)}")
    print(f"config.instead={getattr(config, 'instead', None)}")
    print(f"config.is_half={getattr(config, 'is_half', None)}")

    for env_name in (
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "HF_HOME",
    ):
        print(f"{env_name}={os.getenv(env_name, '')}")

    for device_path in ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia0"):
        print(f"{device_path}.exists={os.path.exists(device_path)}")

    try:
        cuda_available = torch.cuda.is_available()
        print(f"torch.cuda.is_available()={cuda_available}")
    except Exception as exc:
        print(f"torch.cuda.is_available()_error={exc!r}")
        cuda_available = False

    try:
        device_count = torch.cuda.device_count()
        print(f"torch.cuda.device_count()={device_count}")
    except Exception as exc:
        print(f"torch.cuda.device_count()_error={exc!r}")
        device_count = 0

    if not cuda_available and device_count > 0:
        print(
            "NOTE: device_count > 0 while cuda.is_available() is False. "
            "This usually means the driver is partially visible but CUDA initialization is failing."
        )

    for index in range(device_count):
        try:
            props = torch.cuda.get_device_properties(index)
            print(f"device[{index}].name={props.name}")
            print(
                "device[%d].total_memory_gb=%.2f"
                % (index, props.total_memory / 1024 / 1024 / 1024)
            )
        except Exception as exc:
            print(f"device[{index}].probe_error={exc!r}")
            traceback.print_exc()

    print("=== End RVC CUDA Debug ===")


tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets", "weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
if hasattr(torch.backends, "nnpack"):
    torch.backends.nnpack.set_flags(False)
torch.manual_seed(114514)


config = Config()
if config.debug_cuda:
    print_cuda_debug_report(config)
vc = VC(config)

if not config.nocheck:
    from infer.lib.rvcmd import check_all_assets, download_all_assets

    if not check_all_assets(update=config.update):
        if config.update:
            download_all_assets(tmpdir=tmp)
            if not check_all_assets(update=config.update):
                logging.error("counld not satisfy all assets needed.")
                exit(1)

if config.dml == True:

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res

    import fairseq

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml

i18n = I18nAuto()
logger.info(i18n)
# Check if there are NVIDIA GPUs available for training and accelerated inference
gpu_infos = []
mem = []
if_gpu_ok = False

if isinstance(config.device, str) and config.device.startswith("cuda"):
    try:
        ngpu = torch.cuda.device_count()
        if torch.cuda.is_available() and ngpu > 0:
            for i in range(ngpu):
                gpu_name = torch.cuda.get_device_name(i)
                if any(
                    value in gpu_name.upper()
                    for value in [
                        "10",
                        "16",
                        "20",
                        "30",
                        "40",
                        "A2",
                        "A3",
                        "A4",
                        "P4",
                        "A50",
                        "500",
                        "A60",
                        "70",
                        "80",
                        "90",
                        "M4",
                        "T4",
                        "TITAN",
                        "4060",
                        "L",
                        "6000",
                    ]
                ):
                    # A10#A100#V100#A40#P40#M40#K80#A4500
                    if_gpu_ok = True  # At least one NVIDIA GPU is available
                    gpu_infos.append("%s\t%s" % (i, gpu_name))
                    mem.append(
                        int(
                            torch.cuda.get_device_properties(i).total_memory
                            / 1024
                            / 1024
                            / 1024
                            + 0.4
                        )
                    )
    except Exception as exc:
        if config.debug_cuda:
            print(f"startup_gpu_probe_error={exc!r}")
            traceback.print_exc()
        logger.warning("Skipping CUDA GPU probe and falling back to CPU metadata: %s", exc)
if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = i18n(
        "Unfortunately, there is no compatible GPU available to support your training."
    )
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])


weight_root = os.getenv("weight_root")
weight_uvr5_root = os.getenv("weight_uvr5_root")
index_root = os.getenv("index_root")
outside_index_root = os.getenv("outside_index_root")

names = [""]
index_paths = [""]
MERGE_MODEL_METADATA_CACHE = {}


def lookup_names(weight_root):
    global names
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            names.append(name)


def lookup_indices(index_root):
    global index_paths
    for root, _, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append(str(pathlib.Path(root, name)))


lookup_names(weight_root)
lookup_indices(index_root)
lookup_indices(outside_index_root)
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))

AUTO_SEPARATION_MODELS = [
    name
    for name in sorted(uvr5_names)
    if not name.startswith("VR-") and name != "onnx_dereverb_By_FoxJoy"
]
DEFAULT_AUTO_SEPARATION_MODEL = next(
    (
        candidate
        for candidate in ("HP3_all_vocals", "HP2_all_vocals", "HP5_only_main_vocal")
        if candidate in AUTO_SEPARATION_MODELS
    ),
    AUTO_SEPARATION_MODELS[0] if AUTO_SEPARATION_MODELS else "",
)

DEFAULT_GENERATION_OUTPUT_DIR = pathlib.Path(now_dir) / "outputs"
AUDIO_FILE_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".aifc",
    ".caf",
    ".flac",
    ".m4a",
    ".mka",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
    ".wma",
}
TRANSPOSE_MIN = -24
TRANSPOSE_MAX = 24
MAX_MERGED_INDEX_VECTORS = 200000


def _normalize_transpose_value(value, default=0):
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        normalized = int(default)
    return min(TRANSPOSE_MAX, max(TRANSPOSE_MIN, normalized))


def _print_gradio_startup_urls(local_url, share_url):
    if local_url:
        print(f"Gradio started: {local_url}", flush=True)
    if share_url:
        print(f"Gradio public link: {share_url}", flush=True)

INFER_PRESET_MANAGER = PresetManager(pathlib.Path(now_dir, "presets"))
INFER_PRESET_TAB = "infer"
INFER_PRESET_FIELDS = [
    "sid0",
    "spk_item",
    "vc_transform0",
    "file_index2",
    "f0method0",
    "resample_sr0",
    "rms_mix_rate0",
    "protect0",
    "filter_radius0",
    "index_rate1",
    "auto_separate_audio",
    "auto_separate_model",
    "loop_all_models",
    "save_as_mp3",
    "mp3_bitrate_kbps",
    "vc_transform1",
    "dir_input",
    "opt_input",
    "file_index4",
    "f0method1",
    "resample_sr1",
    "rms_mix_rate1",
    "protect1",
    "filter_radius1",
    "index_rate2",
    "format1",
]


def get_infer_preset_defaults():
    return {
        "sid0": "2Pac_Tupac_Demo.pth",
        "spk_item": 0,
        "vc_transform0": 12,
        "file_index2": "",
        "f0method0": "rmvpe",
        "resample_sr0": 0,
        "rms_mix_rate0": 0.25,
        "protect0": 0.33,
        "filter_radius0": 3,
        "index_rate1": 1.0,
        "auto_separate_audio": True,
        "auto_separate_model": DEFAULT_AUTO_SEPARATION_MODEL,
        "loop_all_models": False,
        "save_as_mp3": True,
        "mp3_bitrate_kbps": 192,
        "vc_transform1": 12,
        "dir_input": "",
        "opt_input": "",
        "file_index4": "",
        "f0method1": "rmvpe",
        "resample_sr1": 0,
        "rms_mix_rate1": 0.25,
        "protect1": 0.33,
        "filter_radius1": 3,
        "index_rate2": 1.0,
        "format1": "wav",
    }


def merge_infer_preset_with_defaults(preset_data):
    defaults = get_infer_preset_defaults()
    merged = INFER_PRESET_MANAGER.merge_config(defaults, preset_data or {})
    merged["sid0"] = merged["sid0"] if merged["sid0"] in names else ""
    merged["file_index2"] = (
        merged["file_index2"] if merged["file_index2"] in index_paths else ""
    )
    merged["file_index4"] = (
        merged["file_index4"] if merged["file_index4"] in index_paths else ""
    )
    merged["auto_separate_model"] = (
        merged["auto_separate_model"]
        if merged["auto_separate_model"] in AUTO_SEPARATION_MODELS
        else DEFAULT_AUTO_SEPARATION_MODEL
    )
    merged["mp3_bitrate_kbps"] = min(
        320, max(32, int(merged.get("mp3_bitrate_kbps", 192) or 192))
    )
    merged["vc_transform0"] = _normalize_transpose_value(
        merged.get("vc_transform0"), defaults["vc_transform0"]
    )
    merged["vc_transform1"] = _normalize_transpose_value(
        merged.get("vc_transform1"), defaults["vc_transform1"]
    )
    merged["format1"] = (
        merged["format1"]
        if merged["format1"] in {"wav", "flac", "mp3", "m4a"}
        else "wav"
    )
    return merged


def build_infer_preset_payload(*values):
    payload = {}
    for field_name, value in zip(INFER_PRESET_FIELDS, values):
        payload[field_name] = value
    return merge_infer_preset_with_defaults(payload)


def get_infer_preset_choices():
    return INFER_PRESET_MANAGER.list_presets(INFER_PRESET_TAB, None)


def _normalize_infer_preset_name(preset_name):
    normalized = str(preset_name or "").strip()
    if normalized and normalized in get_infer_preset_choices():
        return normalized
    return None


def load_initial_infer_preset():
    loaded = INFER_PRESET_MANAGER.load_last_used(INFER_PRESET_TAB, None)
    return merge_infer_preset_with_defaults(loaded)


def _strip_path_value(path_value):
    return str(path_value or "").strip().strip('"').strip()


def _resolve_input_dir(input_dir):
    input_value = _strip_path_value(input_dir)
    if not input_value:
        raise ValueError("Input folder is required.")
    path = pathlib.Path(input_value).expanduser()
    if not path.is_absolute():
        path = pathlib.Path(now_dir) / path
    path = path.resolve()
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Input folder not found: {path}")
    return path


def _resolve_output_dir(output_dir=None, require_provided=False):
    output_value = _strip_path_value(output_dir)
    if not output_value:
        if require_provided:
            raise ValueError("Output folder is required.")
        path = DEFAULT_GENERATION_OUTPUT_DIR
    else:
        path = pathlib.Path(output_value).expanduser()
        if not path.is_absolute():
            path = pathlib.Path(now_dir) / path
    path = path.resolve()
    if path.exists() and not path.is_dir():
        raise ValueError(f"Output path is not a folder: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def open_outputs_folder():
    output_dir = _resolve_output_dir()
    output_dir_str = str(output_dir)
    try:
        if os.name == "nt":
            os.startfile(output_dir_str)
        elif platform.system() == "Linux":
            Popen(["xdg-open", output_dir_str])
        elif platform.system() == "Darwin":
            Popen(["open", output_dir_str])
        else:
            raise RuntimeError(f"Unsupported platform: {platform.system()}")
        return
    except Exception as exc:
        raise gr.Error(f"Failed to open outputs folder: {exc}")


def _next_numbered_output_name(output_dir):
    max_number = 0
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        stem = path.stem.split("__", 1)[0]
        if stem.isdigit():
            max_number = max(max_number, int(stem))
    return f"{max_number + 1:04d}"


def _sanitize_output_fragment(text):
    base_text = pathlib.Path(str(text or "model")).stem
    sanitized = "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
        for ch in base_text
    ).strip("._")
    return sanitized or "model"


def _append_model_name_to_output_base(base_name, model_name, append_model_name):
    if not append_model_name:
        return base_name
    return f"{base_name}__{_sanitize_output_fragment(model_name)}"


def _available_model_names():
    return sorted(name for name in names if str(name).strip())


def _refresh_model_name_cache():
    global names
    names = [""]
    lookup_names(weight_root)
    MERGE_MODEL_METADATA_CACHE.clear()
    return sorted(names)


def _refresh_index_path_cache():
    global index_paths
    index_paths = [""]
    lookup_indices(index_root)
    lookup_indices(outside_index_root)
    return sorted(index_paths)


def _normalize_index_path_value(path_value):
    normalized = _strip_path_value(path_value)
    if not normalized:
        return ""
    return str(pathlib.Path(normalized))


def _build_index_dropdown_update(index_update=None, current_value="", index_choices=None):
    if index_choices is None:
        index_choices = _refresh_index_path_cache()
    normalized_choice_map = {
        _normalize_index_path_value(choice): choice for choice in index_choices
    }
    update_payload = {}
    if isinstance(index_update, dict):
        update_payload = {
            key: value for key, value in index_update.items() if key != "__type__"
        }
    requested_value = normalized_choice_map.get(
        _normalize_index_path_value(update_payload.get("value", "")),
        "",
    )
    fallback_value = normalized_choice_map.get(
        _normalize_index_path_value(current_value),
        "",
    )
    selected_value = requested_value or fallback_value or ""
    update_payload["choices"] = index_choices
    update_payload["value"] = selected_value
    return gr.update(**update_payload)


def refresh_infer_model_and_index_choices(selected_model_name, selected_index_path):
    model_choices = _refresh_model_name_cache()
    index_choices = _refresh_index_path_cache()
    selected_model_name = str(selected_model_name or "").strip()
    return (
        gr.update(
            choices=model_choices,
            value=selected_model_name if selected_model_name in model_choices else "",
        ),
        _build_index_dropdown_update(
            {"value": selected_index_path},
            current_value=selected_index_path,
            index_choices=index_choices,
        ),
    )


def refresh_batch_index_choices(selected_index_path):
    return _build_index_dropdown_update(
        {"value": selected_index_path},
        current_value=selected_index_path,
    )


def _resolve_model_names_for_conversion(selected_model_name, loop_all_models):
    available_models = _available_model_names()
    if loop_all_models:
        if not available_models:
            raise ValueError("No voice models were found in assets/weights.")
        return available_models
    selected_model_name = str(selected_model_name or "").strip()
    if not selected_model_name:
        raise ValueError(
            "Please select a voice model in 'Inferencing voice' before converting."
        )
    return [selected_model_name]


def _normalize_mp3_bitrate_kbps(value):
    try:
        bitrate_kbps = int(round(float(value)))
    except (TypeError, ValueError):
        bitrate_kbps = 192
    return min(320, max(32, bitrate_kbps))


def _resolve_output_audio_format(save_as_mp3):
    return "mp3" if save_as_mp3 else "wav"


def toggle_mp3_bitrate_visibility(save_as_mp3):
    return gr.update(visible=bool(save_as_mp3))


def _resolve_selected_voice_model_path(model_name):
    model_name = str(model_name or "").strip()
    if not model_name:
        raise ValueError("Please select a voice model.")
    model_path = pathlib.Path(weight_root) / model_name
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Voice model '{model_name}' was not found in assets/weights."
        )
    return model_name, str(model_path)


def _normalize_model_sr_label(value, config=None):
    sr_lookup = {
        32000: "32k",
        40000: "40k",
        48000: "48k",
    }
    if isinstance(value, str):
        value = value.strip()
        if value in sr_lookup.values():
            return value
    try:
        sr_value = int(value)
    except (TypeError, ValueError):
        sr_value = None
    if sr_value in sr_lookup:
        return sr_lookup[sr_value]
    if isinstance(config, (list, tuple)) and config:
        try:
            config_sr = int(config[-1])
        except (TypeError, ValueError):
            config_sr = None
        if config_sr in sr_lookup:
            return sr_lookup[config_sr]
    return None


def _normalize_model_f0_flag(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes"}:
            return 1
        if normalized in {"0", "false", "no"}:
            return 0
    try:
        return 1 if int(value) else 0
    except (TypeError, ValueError):
        return 1


def _expected_feature_dimension_for_version(version):
    return 256 if str(version or "v1").strip().lower() == "v1" else 768


def _load_merge_model_metadata(model_name):
    model_name, model_path = _resolve_selected_voice_model_path(model_name)
    checkpoint = torch.load(model_path, map_location="cpu")
    config_value = checkpoint.get("config") or []
    sr_label = _normalize_model_sr_label(checkpoint.get("sr"), config_value)
    if not sr_label:
        raise ValueError(f"Could not determine the sample rate for '{model_name}'.")
    version_value = str(checkpoint.get("version") or "v1").strip().lower()
    if version_value not in {"v1", "v2"}:
        version_value = "v1"
    weights = checkpoint["model"] if "model" in checkpoint else checkpoint.get("weight")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"Voice model '{model_name}' does not contain mergeable weights.")
    return {
        "name": model_name,
        "path": model_path,
        "checkpoint": checkpoint,
        "weights": weights,
        "sr": sr_label,
        "f0": _normalize_model_f0_flag(checkpoint.get("f0", 1)),
        "version": version_value,
    }


def _load_cached_merge_model_metadata(model_name, force_refresh=False):
    model_name, model_path = _resolve_selected_voice_model_path(model_name)
    stat_result = os.stat(model_path)
    cache_key = (model_path, stat_result.st_mtime_ns, stat_result.st_size)
    cached = MERGE_MODEL_METADATA_CACHE.get(model_name)
    if not force_refresh and cached and cached.get("cache_key") == cache_key:
        return cached["metadata"]
    metadata = _load_merge_model_metadata(model_name)
    MERGE_MODEL_METADATA_CACHE[model_name] = {
        "cache_key": cache_key,
        "metadata": metadata,
    }
    return metadata


def _load_merge_index_metadata(model_name, expected_dimension):
    index_path = _strip_path_value(get_index_path_from_model(model_name))
    if not index_path:
        return None
    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            f"Feature index for '{model_name}' was not found at '{index_path}'."
        )
    index = faiss.read_index(index_path)
    if int(index.d) != int(expected_dimension):
        raise ValueError(
            f"Feature index for '{model_name}' has dimension {index.d}, "
            f"expected {expected_dimension}."
        )
    return {
        "name": model_name,
        "path": index_path,
        "index": index,
        "dimension": int(index.d),
        "ntotal": int(index.ntotal),
    }


def _validate_mergeable_voice_models(model_a, model_b):
    if model_a["sr"] != model_b["sr"]:
        raise ValueError(
            "Selected voice models use different sample rates "
            f"({model_a['sr']} vs {model_b['sr']})."
        )
    if model_a["f0"] != model_b["f0"]:
        raise ValueError(
            "Selected voice models disagree on pitch-guidance support "
            f"({model_a['f0']} vs {model_b['f0']})."
        )
    if model_a["version"] != model_b["version"]:
        raise ValueError(
            "Selected voice models use different architecture versions "
            f"({model_a['version']} vs {model_b['version']})."
        )

    weights_a = model_a["weights"]
    weights_b = model_b["weights"]
    if sorted(weights_a.keys()) != sorted(weights_b.keys()):
        raise ValueError(
            "Selected voice models use different parameter sets and cannot be merged."
        )

    for key in weights_a.keys():
        shape_a = tuple(weights_a[key].shape)
        shape_b = tuple(weights_b[key].shape)
        if key == "emb_g.weight":
            if shape_a[1:] != shape_b[1:]:
                raise ValueError(
                    "Selected voice models have incompatible speaker embedding shapes "
                    f"for '{key}': {shape_a} vs {shape_b}."
                )
            continue
        if shape_a != shape_b:
            raise ValueError(
                f"Selected voice models have incompatible tensor shapes for '{key}': "
                f"{shape_a} vs {shape_b}."
            )


def _normalize_merge_model_selection(model_names):
    if model_names is None:
        return []
    if isinstance(model_names, (str, bytes)):
        model_names = [model_names]
    normalized = []
    seen = set()
    for model_name in model_names:
        normalized_name = str(model_name or "").strip()
        if not normalized_name:
            continue
        if normalized_name in seen:
            raise ValueError(
                f"Voice model '{normalized_name}' was selected more than once."
            )
        seen.add(normalized_name)
        normalized.append(normalized_name)
    return normalized


def _parse_merge_weights(weight_text, model_count):
    model_count = int(model_count)
    if model_count < 2:
        raise ValueError("Please select at least two voice models to merge.")
    raw_text = str(weight_text or "").strip()
    if not raw_text:
        return [1.0 / model_count] * model_count

    tokens = [
        token
        for token in re.split(r"[\s,;]+", raw_text.replace("\n", " ").strip())
        if token
    ]
    if len(tokens) != model_count:
        raise ValueError(
            f"Expected {model_count} weight values, but received {len(tokens)}."
        )

    weights = []
    for token in tokens:
        is_percent = token.endswith("%")
        numeric_token = token[:-1] if is_percent else token
        try:
            value = float(numeric_token)
        except ValueError as exc:
            raise ValueError(f"Invalid weight value '{token}'.") from exc
        if is_percent:
            value /= 100.0
        if value < 0:
            raise ValueError("Weights must be non-negative.")
        weights.append(value)

    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("At least one merge weight must be greater than zero.")
    return [value / total_weight for value in weights]


def _validate_mergeable_voice_model_group(models):
    if len(models) < 2:
        raise ValueError("Please select at least two voice models to merge.")
    reference_model = models[0]
    for current_model in models[1:]:
        _validate_mergeable_voice_models(reference_model, current_model)


def _calculate_weighted_index_sample_counts(totals, weights, max_vectors):
    totals = [max(0, int(total)) for total in totals]
    max_vectors = max(1, int(max_vectors))
    if len(totals) != len(weights):
        raise ValueError("Index totals and weights do not have the same length.")
    combined_total = sum(totals)
    if combined_total <= max_vectors:
        return totals

    normalized_weights = [float(weight) for weight in weights]
    samples = [
        min(total, int(round(max_vectors * weight)))
        for total, weight in zip(totals, normalized_weights)
    ]
    for index, total in enumerate(totals):
        if total > 0 and samples[index] == 0:
            samples[index] = 1

    while sum(samples) > max_vectors:
        candidate_indices = [
            index for index, sample in enumerate(samples) if sample > 0
        ]
        if not candidate_indices:
            break
        reduce_index = max(
            candidate_indices,
            key=lambda index: (
                samples[index] - max_vectors * normalized_weights[index],
                samples[index],
            ),
        )
        samples[reduce_index] -= 1

    while sum(samples) < max_vectors:
        candidate_indices = [
            index
            for index, total in enumerate(totals)
            if samples[index] < total
        ]
        if not candidate_indices:
            break
        add_index = max(
            candidate_indices,
            key=lambda index: (
                max_vectors * normalized_weights[index] - samples[index],
                totals[index] - samples[index],
            ),
        )
        samples[add_index] += 1

    return samples


def _sample_vectors_from_index(index, sample_size, rng):
    sample_size = max(0, int(sample_size))
    if sample_size == 0 or int(index.ntotal) == 0:
        return np.empty((0, int(index.d)), dtype=np.float32)
    if sample_size >= int(index.ntotal):
        return np.asarray(index.reconstruct_n(0, int(index.ntotal)), dtype=np.float32)

    index.make_direct_map()
    sampled_ids = np.asarray(
        rng.choice(int(index.ntotal), size=sample_size, replace=False),
        dtype=np.int64,
    )
    return np.asarray(index.reconstruct_batch(sampled_ids), dtype=np.float32)


def create_merged_feature_index(
    model_names,
    merged_model_name,
    normalized_weights,
    expected_dimension,
):
    index_metas = [
        _load_merge_index_metadata(model_name, expected_dimension)
        for model_name in model_names
    ]
    missing = [
        model_name
        for model_name, index_meta in zip(model_names, index_metas)
        if index_meta is None
    ]
    if missing:
        missing_list = ", ".join(missing)
        return (
            None,
            f"Skipped merged index because no source index was found for: {missing_list}.",
        )

    sample_counts = _calculate_weighted_index_sample_counts(
        [index_meta["ntotal"] for index_meta in index_metas],
        normalized_weights,
        MAX_MERGED_INDEX_VECTORS,
    )
    if sum(sample_counts) <= 0:
        return (
            None,
            "Skipped merged index because the selected source indices do not contain vectors.",
        )

    rng = np.random.default_rng(114514)
    vector_chunks = [
        _sample_vectors_from_index(index_meta["index"], sample_count, rng)
        for index_meta, sample_count in zip(index_metas, sample_counts)
    ]
    merged_vectors = np.concatenate(vector_chunks, axis=0).astype(
        np.float32, copy=False
    )
    if merged_vectors.shape[0] == 0:
        return (
            None,
            "Skipped merged index because no source vectors were available after sampling.",
        )

    rng.shuffle(merged_vectors, axis=0)
    n_ivf = min(
        int(16 * np.sqrt(merged_vectors.shape[0])),
        max(1, merged_vectors.shape[0] // 39),
    )
    n_ivf = max(1, n_ivf)
    merged_index = faiss.index_factory(
        int(expected_dimension), f"IVF{n_ivf},Flat"
    )
    index_ivf = faiss.extract_index_ivf(merged_index)
    index_ivf.nprobe = 1
    merged_index.train(merged_vectors)
    batch_size_add = 8192
    for i in range(0, merged_vectors.shape[0], batch_size_add):
        merged_index.add(merged_vectors[i : i + batch_size_add])

    outside_index_dir = pathlib.Path(outside_index_root)
    outside_index_dir.mkdir(parents=True, exist_ok=True)
    merged_index_path = str(outside_index_dir / f"{merged_model_name}.index")
    faiss.write_index(merged_index, merged_index_path)
    sample_summary = ", ".join(
        f"{sample_count} from {model_name}"
        for model_name, sample_count in zip(model_names, sample_counts)
    )
    info = (
        f"Created merged index '{merged_index_path}' using {sample_summary}."
    )
    return merged_index_path, info


def _suggest_merged_voice_model_name(model_names):
    normalized_names = [
        _sanitize_output_fragment(model_name)
        for model_name in _normalize_merge_model_selection(model_names)
    ]
    if len(normalized_names) < 2:
        return ""
    name_parts = normalized_names[:3]
    if len(normalized_names) > 3:
        name_parts.append(f"plus_{len(normalized_names) - 3}_more")
    return "__".join(name_parts) + f"__mix_{len(normalized_names):02d}"


def suggest_merged_voice_model_name(selected_models):
    return _suggest_merged_voice_model_name(selected_models)


def suggest_merged_voice_model_name_from_merge_inputs(base_model_name, selected_models):
    return _suggest_merged_voice_model_name(
        _build_full_merge_model_selection(base_model_name, selected_models)
    )


def _build_merged_voice_info(model_names, normalized_weights, custom_info):
    custom_info = str(custom_info or "").strip()
    if custom_info:
        return custom_info
    parts = [
        f"{model_name} ({float(weight) * 100:.1f}%)"
        for model_name, weight in zip(model_names, normalized_weights)
    ]
    return "Merged voice created from " + ", ".join(parts) + "."


def refresh_merge_model_dropdowns(selected_models):
    _refresh_model_name_cache()
    available_models = _available_model_names()
    selected_models = _normalize_merge_model_selection(selected_models)
    filtered_selection = [
        model_name for model_name in selected_models if model_name in available_models
    ]
    if len(filtered_selection) < 2 and len(available_models) >= 2:
        filtered_selection = available_models[:2]
    return gr.update(choices=available_models, value=filtered_selection)


def _short_merge_compatibility_reason(reason):
    reason = str(reason or "").strip()
    if "sample rates" in reason:
        return "sample rate mismatch"
    if "pitch-guidance support" in reason:
        return "pitch-guidance mismatch"
    if "architecture versions" in reason:
        return "architecture version mismatch"
    if "parameter sets" in reason:
        return "parameter set mismatch"
    if "speaker embedding shapes" in reason:
        return "speaker embedding shape mismatch"
    if "tensor shapes" in reason:
        return "tensor shape mismatch"
    return reason or "incompatible"


def _build_full_merge_model_selection(base_model_name, selected_models):
    base_model_name = str(base_model_name or "").strip()
    selected_models = _normalize_merge_model_selection(selected_models)
    if base_model_name:
        selected_models = [
            model_name for model_name in selected_models if model_name != base_model_name
        ]
        return [base_model_name] + selected_models
    return selected_models


def _format_merge_compatibility_status(base_model_name, compatible_models, incompatible):
    total_models = len(compatible_models) + len(incompatible)
    additional_compatible = max(0, len(compatible_models) - (1 if base_model_name else 0))
    if not base_model_name:
        return "Select a base voice model to scan merge compatibility."
    lines = [
        f"Scanned {total_models} voice models against {base_model_name}.",
        f"Base voice is included automatically: {base_model_name}",
        f"Additional compatible voices: {additional_compatible}",
        f"Filtered out: {len(incompatible)}",
    ]
    if incompatible:
        reason_counts = Counter(
            _short_merge_compatibility_reason(reason) for _, reason in incompatible
        )
        lines.append(
            "Reason summary: "
            + ", ".join(
                f"{reason} x{count}" for reason, count in reason_counts.most_common(4)
            )
        )
        preview_lines = [
            f"{model_name}: {_short_merge_compatibility_reason(reason)}"
            for model_name, reason in incompatible[:10]
        ]
        if preview_lines:
            lines.append("Filtered examples:")
            lines.extend(preview_lines)
    else:
        lines.append("All scanned models are compatible.")
    return "\n".join(lines)


def _format_incompatible_merge_voice_list(incompatible):
    if not incompatible:
        return "None."
    return "\n".join(
        f"{model_name}: {_short_merge_compatibility_reason(reason)}"
        for model_name, reason in incompatible
    )


def _scan_merge_compatible_models(base_model_name, available_models=None, progress=None):
    available_models = available_models or _available_model_names()
    if not available_models:
        return "", [], "No voice models were found in assets/weights."

    base_model_name = str(base_model_name or "").strip()
    if not base_model_name or base_model_name not in available_models:
        base_model_name = available_models[0]

    if progress is not None:
        progress(0, desc=f"Scanning merge compatibility for {base_model_name}")

    base_model = _load_cached_merge_model_metadata(base_model_name)
    compatible_models = []
    incompatible = []
    total = max(1, len(available_models))
    for index, candidate_name in enumerate(available_models, start=1):
        if progress is not None:
            progress(
                index / total,
                desc=f"Scanning {index}/{total}: {candidate_name}",
            )
        try:
            candidate_model = _load_cached_merge_model_metadata(candidate_name)
            _validate_mergeable_voice_models(base_model, candidate_model)
            compatible_models.append(candidate_name)
        except Exception as exc:
            incompatible.append((candidate_name, str(exc)))

    if base_model_name in compatible_models:
        compatible_models = [base_model_name] + [
            model_name for model_name in compatible_models if model_name != base_model_name
        ]
    else:
        compatible_models.insert(0, base_model_name)
    status = _format_merge_compatibility_status(
        base_model_name,
        compatible_models,
        incompatible,
    )
    incompatible_text = _format_incompatible_merge_voice_list(incompatible)
    return base_model_name, compatible_models, status, incompatible_text


def _build_merge_scan_updates(
    available_models,
    base_model_name,
    compatible_models,
    selected_models,
    status_text,
):
    selectable_compatible_models = [
        model_name for model_name in compatible_models if model_name != base_model_name
    ]
    selected_models = _normalize_merge_model_selection(selected_models)
    filtered_selection = [
        model_name
        for model_name in selected_models
        if model_name in selectable_compatible_models
    ]
    suggested_name = _suggest_merged_voice_model_name(
        _build_full_merge_model_selection(base_model_name, filtered_selection)
    )
    return (
        gr.update(choices=available_models, value=base_model_name),
        gr.update(choices=selectable_compatible_models, value=filtered_selection),
        suggested_name,
        status_text,
    )


def scan_merge_compatibility(base_model_name, selected_models, progress=gr.Progress(track_tqdm=False)):
    available_models = _available_model_names()
    (
        base_model_name,
        compatible_models,
        status,
        incompatible_text,
    ) = _scan_merge_compatible_models(
        base_model_name,
        available_models=available_models,
        progress=progress,
    )
    selected_models = _normalize_merge_model_selection(selected_models)
    filtered_selection = [
        model_name
        for model_name in selected_models
        if model_name in compatible_models and model_name != base_model_name
    ]
    selectable_compatible_models = [
        model_name for model_name in compatible_models if model_name != base_model_name
    ]
    suggested_name = _suggest_merged_voice_model_name(
        _build_full_merge_model_selection(base_model_name, filtered_selection)
    )
    return (
        gr.update(choices=available_models, value=base_model_name),
        gr.update(choices=selectable_compatible_models, value=filtered_selection),
        suggested_name,
        status,
    )


def stream_scan_merge_compatibility(base_model_name, selected_models):
    available_models = _available_model_names()
    if not available_models:
        status_text = "No voice models were found in assets/weights."
        yield _build_merge_scan_updates(
            [],
            "",
            [],
            [],
            status_text,
        )
        return

    base_model_name = str(base_model_name or "").strip()
    if not base_model_name or base_model_name not in available_models:
        base_model_name = available_models[0]

    _emit_progress_line(f"Starting merge compatibility scan for {base_model_name}.")
    base_model = _load_cached_merge_model_metadata(base_model_name)
    compatible_models = [base_model_name]
    incompatible = []
    selected_models = _normalize_merge_model_selection(selected_models)
    total = max(1, len(available_models))

    initial_status = "\n".join(
        [
            f"Scanning {total} voice models against {base_model_name}...",
            "Base voice is included automatically.",
            "Finding additional compatible voices...",
        ]
    )
    yield _build_merge_scan_updates(
        available_models,
        base_model_name,
        compatible_models,
        selected_models,
        initial_status,
    )

    for index, candidate_name in enumerate(available_models, start=1):
        if candidate_name == base_model_name:
            status_lines = [
                f"Scanning {index}/{total}: {candidate_name}",
                f"Compatible so far: {max(0, len(compatible_models) - 1)} additional voices",
                f"Filtered so far: {len(incompatible)}",
            ]
            _emit_progress_line(
                f"Merge scan {index}/{total}: {candidate_name} -> base voice"
            )
            yield _build_merge_scan_updates(
                available_models,
                base_model_name,
                compatible_models,
                selected_models,
                "\n".join(status_lines),
            )
            continue

        try:
            candidate_model = _load_cached_merge_model_metadata(candidate_name)
            _validate_mergeable_voice_models(base_model, candidate_model)
            compatible_models.append(candidate_name)
            result_label = "compatible"
        except Exception as exc:
            incompatible.append((candidate_name, str(exc)))
            result_label = _short_merge_compatibility_reason(exc)

        status_lines = [
            f"Scanning {index}/{total}: {candidate_name}",
            f"Compatible so far: {max(0, len(compatible_models) - 1)} additional voices",
            f"Filtered so far: {len(incompatible)}",
        ]
        _emit_progress_line(
            f"Merge scan {index}/{total}: {candidate_name} -> {result_label}"
        )
        yield _build_merge_scan_updates(
            available_models,
            base_model_name,
            compatible_models,
            selected_models,
            "\n".join(status_lines),
        )

    final_status = _format_merge_compatibility_status(
        base_model_name,
        compatible_models,
        incompatible,
    )
    _emit_progress_line(
        "Merge compatibility scan finished: "
        f"{max(0, len(compatible_models) - 1)} additional compatible, "
        f"{len(incompatible)} filtered."
    )
    yield _build_merge_scan_updates(
        available_models,
        base_model_name,
        compatible_models,
        selected_models,
        final_status,
    )


def stream_refresh_merge_compatibility_inputs(base_model_name, selected_models):
    _refresh_model_name_cache()
    yield from stream_scan_merge_compatibility(base_model_name, selected_models)


def refresh_merge_compatibility_inputs(base_model_name, selected_models, progress=gr.Progress(track_tqdm=False)):
    available_models = _refresh_model_name_cache()
    (
        base_model_name,
        compatible_models,
        status,
        _incompatible_text,
    ) = _scan_merge_compatible_models(
        base_model_name,
        available_models=available_models,
        progress=progress,
    )
    selected_models = _normalize_merge_model_selection(selected_models)
    filtered_selection = [
        model_name
        for model_name in selected_models
        if model_name in compatible_models and model_name != base_model_name
    ]
    selectable_compatible_models = [
        model_name for model_name in compatible_models if model_name != base_model_name
    ]
    suggested_name = _suggest_merged_voice_model_name(
        _build_full_merge_model_selection(base_model_name, filtered_selection)
    )
    return (
        gr.update(choices=available_models, value=base_model_name),
        gr.update(choices=selectable_compatible_models, value=filtered_selection),
        suggested_name,
        status,
    )


def _resolve_selected_index_path(file_index, file_index2):
    if file_index:
        if hasattr(file_index, "name"):
            file_index = str(file_index.name)
        file_index = (
            str(file_index)
            .strip(" ")
            .strip('"')
            .strip("\n")
            .strip('"')
            .strip(" ")
            .replace("trained", "added")
        )
        return file_index
    if file_index2:
        return _strip_path_value(file_index2)
    return ""


def _resolve_index_inputs_for_model(model_name, loop_all_models, file_index, file_index2):
    if not loop_all_models:
        return file_index, file_index2
    return "", _strip_path_value(get_index_path_from_model(model_name))


def _save_output_with_metadata(
    output_dir,
    base_name,
    audio,
    sr,
    audio_format,
    bitrate_kbps,
    metadata,
):
    output_dir = pathlib.Path(output_dir)
    audio_path = output_dir / f"{base_name}.{audio_format}"
    metadata_path = output_dir / f"{base_name}.txt"
    save_audio(
        str(audio_path),
        audio,
        sr,
        f32=False,
        format=audio_format,
        bitrate_kbps=bitrate_kbps,
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return str(audio_path), str(metadata_path)


def _build_conversion_metadata(
    mode,
    model_name,
    speaker_id,
    input_audio_path,
    output_audio_path,
    output_metadata_path,
    sample_rate,
    audio_format,
    f0_up_key,
    f0_method,
    index_path,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
    auto_separate_audio,
    auto_separate_model,
    loop_all_models,
    save_as_mp3,
    mp3_bitrate_kbps,
    conversion_info,
):
    return {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "model_name": model_name or "",
        "speaker_id": int(speaker_id),
        "input_audio_path": str(input_audio_path),
        "output_audio_path": str(output_audio_path),
        "output_metadata_path": str(output_metadata_path),
        "sample_rate": int(sample_rate),
        "audio_format": audio_format,
        "transpose": _normalize_transpose_value(f0_up_key),
        "f0_method": f0_method,
        "index_path": index_path,
        "index_rate": float(index_rate),
        "filter_radius": int(filter_radius),
        "resample_sr": int(resample_sr),
        "rms_mix_rate": float(rms_mix_rate),
        "protect": float(protect),
        "auto_separate_audio": bool(auto_separate_audio),
        "auto_separate_model": auto_separate_model if auto_separate_audio else "",
        "loop_all_models": bool(loop_all_models),
        "save_as_mp3": bool(save_as_mp3),
        "mp3_bitrate_kbps": (
            int(mp3_bitrate_kbps) if save_as_mp3 else 0
        ),
        "conversion_info": conversion_info,
    }


def _prepare_inference_input(input_audio_path, auto_separate_audio, auto_separate_model):
    resolved_input_path = input_audio_path
    if resolved_input_path is None:
        raise ValueError("You need to upload an audio")
    if hasattr(resolved_input_path, "name"):
        resolved_input_path = str(resolved_input_path.name)
    else:
        resolved_input_path = str(resolved_input_path)

    prepared = {
        "resolved_input_path": resolved_input_path,
        "conversion_input_path": resolved_input_path,
        "stem_info": None,
    }
    if not auto_separate_audio:
        return prepared

    stem_info = separate_audio_for_auto_merge(resolved_input_path, auto_separate_model)
    prepared["conversion_input_path"] = stem_info["vocal_path"]
    prepared["stem_info"] = stem_info
    return prepared


def _cleanup_prepared_inference_input(prepared_input):
    if not prepared_input:
        return
    stem_info = prepared_input.get("stem_info")
    if stem_info is not None:
        shutil.rmtree(stem_info["work_dir"], ignore_errors=True)


def _run_single_inference_conversion(
    model_name,
    sid,
    input_audio_path,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    file_index2,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
    auto_separate_audio,
    auto_separate_model,
    prepared_input=None,
):
    managed_prepared_input = prepared_input is None
    resolved_input_path = ""
    resolved_index_path = ""
    f0_up_key = _normalize_transpose_value(f0_up_key)
    try:
        prepared_input = prepared_input or _prepare_inference_input(
            input_audio_path, auto_separate_audio, auto_separate_model
        )
        resolved_input_path = prepared_input["resolved_input_path"]
        conversion_input_path = prepared_input["conversion_input_path"]
        resolved_index_path = _resolve_selected_index_path(file_index, file_index2)

        info, opt = vc.vc_single(
            sid,
            conversion_input_path,
            f0_up_key,
            f0_file,
            f0_method,
            file_index,
            file_index2,
            index_rate,
            filter_radius,
            resample_sr,
            rms_mix_rate,
            protect,
        )
        if opt is None:
            return info, None, resolved_input_path, resolved_index_path
        if not auto_separate_audio:
            return info, opt, resolved_input_path, resolved_index_path

        target_sr, converted_audio = opt
        stem_info = prepared_input["stem_info"]
        merged_audio = merge_converted_vocal_with_instrumental(
            stem_info["instrumental_path"],
            stem_info["vocal_path"],
            converted_audio,
            target_sr,
        )
        merge_info = (
            f"Auto split/remix: enabled ({auto_separate_model}).\n"
            "Workflow: separate vocals + accompaniment -> convert vocals -> remix."
        )
        return (
            f"{info}\n{merge_info}",
            (target_sr, merged_audio),
            resolved_input_path,
            resolved_index_path,
        )
    except Exception:
        logger.warning(traceback.format_exc())
        return traceback.format_exc(), None, resolved_input_path, resolved_index_path
    finally:
        if managed_prepared_input:
            _cleanup_prepared_inference_input(prepared_input)


def get_vc_for_infer_ui(sid, protect0, protect1, file_index2, file_index4):
    updates = vc.get_vc(sid, protect0, protect1, file_index2, file_index4)
    if not isinstance(updates, tuple) or len(updates) != 6:
        return updates

    (
        speaker_update,
        protect0_update,
        protect1_update,
        file_index2_update,
        file_index4_update,
        model_info,
    ) = updates
    if isinstance(protect1_update, dict):
        protect1_update = {**protect1_update, "visible": False}
    index_choices = _refresh_index_path_cache()
    file_index2_update = _build_index_dropdown_update(
        file_index2_update,
        current_value=file_index2,
        index_choices=index_choices,
    )
    file_index4_update = _build_index_dropdown_update(
        file_index4_update,
        current_value=file_index4,
        index_choices=index_choices,
    )
    return (
        speaker_update,
        protect0_update,
        protect1_update,
        file_index2_update,
        file_index4_update,
        model_info,
    )


def create_merged_voice_model(
    base_model_name,
    selected_model_names,
    merge_weight_text,
    merged_model_name,
    merged_model_info,
    merge_source_indices,
    protect0,
    protect1,
    file_index2,
    file_index4,
):
    try:
        selected_model_names = _build_full_merge_model_selection(
            base_model_name,
            selected_model_names,
        )
        if len(selected_model_names) < 2:
            raise ValueError(
                "Please select at least one compatible voice in addition to the base voice."
            )
        normalized_weights = _parse_merge_weights(
            merge_weight_text,
            len(selected_model_names),
        )
        merge_models = [
            _load_cached_merge_model_metadata(model_name)
            for model_name in selected_model_names
        ]
        _validate_mergeable_voice_model_group(merge_models)
        reference_model = merge_models[0]

        merged_model_name = _sanitize_output_fragment(
            merged_model_name
            or _suggest_merged_voice_model_name(selected_model_names)
        )
        merged_model_info = _build_merged_voice_info(
            selected_model_names,
            normalized_weights,
            merged_model_info,
        )
        status_lines = []
        merge_result = merge_many(
            [model["path"] for model in merge_models],
            normalized_weights,
            reference_model["sr"],
            i18n("Yes") if reference_model["f0"] else i18n("No"),
            merged_model_info,
            merged_model_name,
            reference_model["version"],
        )
        if str(merge_result).strip() != "Success.":
            raise RuntimeError(str(merge_result).strip() or "Voice merge failed.")
        weight_summary = ", ".join(
            f"{model_name} ({weight * 100:.1f}%)"
            for model_name, weight in zip(selected_model_names, normalized_weights)
        )
        status_lines.append(
            f"Created merged voice '{merged_model_name}.pth' from {weight_summary}."
        )

        if merge_source_indices:
            expected_dimension = _expected_feature_dimension_for_version(
                reference_model["version"]
            )
            _, index_status = create_merged_feature_index(
                selected_model_names,
                merged_model_name,
                normalized_weights,
                expected_dimension,
            )
            status_lines.append(index_status)
        else:
            status_lines.append("Skipped merged index because index merge was disabled.")

        model_dropdown_choices = _refresh_model_name_cache()
        _refresh_index_path_cache()
        available_models = _available_model_names()
        (
            merge_base_model_value,
            compatible_models,
            compatibility_status,
            _incompatible_text,
        ) = _scan_merge_compatible_models(selected_model_names[0], available_models)
        selectable_compatible_models = [
            model_name
            for model_name in compatible_models
            if model_name != merge_base_model_value
        ]
        selected_additional_models = [
            model_name
            for model_name in selected_model_names
            if model_name != merge_base_model_value
        ]
        merged_file_name = f"{merged_model_name}.pth"
        (
            speaker_update,
            protect0_update,
            protect1_update,
            file_index2_update,
            file_index4_update,
            model_info,
        ) = get_vc_for_infer_ui(
            merged_file_name,
            protect0,
            protect1,
            file_index2,
            file_index4,
        )
        status_message = "\n".join(status_lines)
        return (
            status_message,
            gr.update(choices=model_dropdown_choices, value=merged_file_name),
            gr.update(choices=available_models, value=merge_base_model_value),
            gr.update(
                choices=selectable_compatible_models,
                value=selected_additional_models,
            ),
            compatibility_status,
            speaker_update,
            protect0_update,
            protect1_update,
            file_index2_update,
            file_index4_update,
            model_info,
        )
    except Exception as exc:
        logger.warning(traceback.format_exc())
        return (
            f"Could not create merged voice: {exc}",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )


def _normalize_audio_float(audio):
    audio = np.asarray(audio)
    if audio.dtype.kind in {"i", "u"}:
        max_val = float(np.iinfo(audio.dtype).max)
        if max_val > 0:
            audio = audio.astype(np.float32) / max_val
        else:
            audio = audio.astype(np.float32)
    else:
        audio = audio.astype(np.float32)
    return audio


def _to_channels_first(audio):
    audio = _normalize_audio_float(audio)
    if audio.ndim == 1:
        return audio[np.newaxis, :]
    if audio.ndim == 2 and audio.shape[0] <= 8:
        return audio
    if audio.ndim == 2:
        return audio.T
    raise ValueError("Unsupported audio shape")


def _match_channels(audio, target_channels):
    if audio.shape[0] == target_channels:
        return audio
    if audio.shape[0] == 1 and target_channels > 1:
        return np.repeat(audio, target_channels, axis=0)
    if target_channels == 1:
        return audio.mean(axis=0, keepdims=True)
    if audio.shape[0] > target_channels:
        return audio[:target_channels]
    pad = np.repeat(audio[-1:, :], target_channels - audio.shape[0], axis=0)
    return np.concatenate([audio, pad], axis=0)


def _pad_or_trim_audio(audio, target_len):
    current_len = audio.shape[1]
    if current_len == target_len:
        return audio
    if current_len > target_len:
        return audio[:, :target_len]
    pad_width = target_len - current_len
    return np.pad(audio, ((0, 0), (0, pad_width)), mode="constant")


def _rms(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def _match_reference_rms(reference_audio, target_audio):
    reference_rms = _rms(reference_audio)
    target_rms = _rms(target_audio)
    if reference_rms <= 1e-6 or target_rms <= 1e-6:
        return target_audio
    return target_audio * (reference_rms / target_rms)


def _cleanup_uvr_model(pre_fun):
    if pre_fun is None:
        return
    try:
        del pre_fun.model
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def separate_audio_for_auto_merge(input_audio_path, model_name):
    if not model_name:
        raise ValueError("Auto separation model is not configured.")

    model_path = os.path.join(os.getenv("weight_uvr5_root"), model_name + ".pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"UVR model not found: {model_path}")

    work_dir = pathlib.Path(tmp) / "auto_merge" / uuid.uuid4().hex
    vocal_dir = work_dir / "vocals"
    instrumental_dir = work_dir / "instrumentals"
    vocal_dir.mkdir(parents=True, exist_ok=True)
    instrumental_dir.mkdir(parents=True, exist_ok=True)

    reformatted_input = work_dir / "input_44100_stereo.wav"
    working_input = str(input_audio_path)
    pre_fun = None

    try:
        channels, rate = get_audio_properties(input_audio_path)
        if channels != 2 or rate != 44100:
            resample_audio(
                input_audio_path,
                str(reformatted_input),
                "pcm_s16le",
                "s16",
                44100,
                "stereo",
            )
            working_input = str(reformatted_input)

        pre_fun = AudioPre(
            agg=10,
            model_path=model_path,
            device=config.device,
            is_half=config.is_half,
        )
        pre_fun._path_audio_(working_input, str(instrumental_dir), str(vocal_dir), "wav")

        created_files = list(vocal_dir.glob("*.wav")) + list(instrumental_dir.glob("*.wav"))
        vocal_path = next(
            (str(path) for path in created_files if path.name.startswith("vocal_")),
            None,
        )
        instrumental_path = next(
            (str(path) for path in created_files if path.name.startswith("instrument_")),
            None,
        )

        if vocal_path is None or instrumental_path is None:
            raise RuntimeError("UVR did not produce both vocal and instrumental stems.")

        return {
            "work_dir": str(work_dir),
            "vocal_path": vocal_path,
            "instrumental_path": instrumental_path,
        }
    finally:
        _cleanup_uvr_model(pre_fun)


def merge_converted_vocal_with_instrumental(instrumental_path, reference_vocal_path, converted_audio, sr):
    instrumental_audio = _to_channels_first(load_audio(instrumental_path, sr, mono=False))
    reference_vocal = _to_channels_first(load_audio(reference_vocal_path, sr, mono=False))
    converted_vocal = _to_channels_first(converted_audio)

    target_channels = instrumental_audio.shape[0]
    target_len = instrumental_audio.shape[1]

    reference_vocal = _match_channels(reference_vocal, target_channels)
    converted_vocal = _match_channels(converted_vocal, target_channels)

    reference_vocal = _pad_or_trim_audio(reference_vocal, target_len)
    converted_vocal = _pad_or_trim_audio(converted_vocal, target_len)

    converted_vocal = _match_reference_rms(reference_vocal, converted_vocal)
    mixed = instrumental_audio + converted_vocal

    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 0.99:
        mixed = mixed / peak * 0.99

    mixed = np.clip(mixed, -1.0, 1.0)
    mixed_int16 = (mixed * 32767.0).astype(np.int16)
    if mixed_int16.shape[0] == 1:
        return mixed_int16[0]
    return mixed_int16.T


def _load_model_for_conversion(model_name, protect):
    vc.get_vc(model_name, protect, protect, "", "")


def _restore_model_after_loop(selected_model_name, protect, file_index2):
    try:
        selected_model_name = str(selected_model_name or "").strip()
        if selected_model_name:
            vc.get_vc(selected_model_name, protect, protect, file_index2, file_index2)
        else:
            vc.get_vc("", protect, protect, "", "")
    except Exception:
        logger.warning(traceback.format_exc())


def _emit_progress_line(message):
    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {message}",
        flush=True,
    )


def _build_progress_output(status_lines, info_blocks):
    sections = []
    if status_lines:
        sections.append("\n".join(status_lines))
    if info_blocks:
        sections.append("\n\n".join(info_blocks))
    return "\n\n".join(section for section in sections if section).strip()


def infer_convert_single(
    model_name,
    sid,
    input_audio_path,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    file_index2,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
    auto_separate_audio,
    auto_separate_model,
    loop_all_models,
    save_as_mp3,
    mp3_bitrate_kbps,
    progress=gr.Progress(track_tqdm=False),
):
    prepared_input = None
    status_lines = []
    info_blocks = []
    preview_result = None
    preview_model = ""
    f0_up_key = _normalize_transpose_value(f0_up_key)
    try:
        model_names = _resolve_model_names_for_conversion(model_name, loop_all_models)
        prepared_input = _prepare_inference_input(
            input_audio_path, auto_separate_audio, auto_separate_model
        )
    except Exception:
        logger.warning(traceback.format_exc())
        failure_info = traceback.format_exc()
        _emit_progress_line("Single conversion failed during preparation.")
        yield failure_info, None
        return

    output_dir = _resolve_output_dir()
    base_name = _next_numbered_output_name(output_dir)
    audio_format = _resolve_output_audio_format(save_as_mp3)
    bitrate_kbps = _normalize_mp3_bitrate_kbps(mp3_bitrate_kbps)
    preview_target_model = model_name if model_name in model_names else model_names[0]
    total_models = len(model_names)
    start_message = (
        f"Starting single conversion across {total_models} model(s)."
        if loop_all_models
        else f"Starting single conversion for model {preview_target_model}."
    )
    status_lines.append(start_message)
    _emit_progress_line(start_message)
    progress(0, desc=start_message[:120])
    yield _build_progress_output(status_lines, info_blocks), None

    try:
        for model_index, current_model_name in enumerate(model_names, start=1):
            model_header = f"[{current_model_name}]"
            model_start_message = (
                f"[{model_index}/{total_models}] Starting model {current_model_name}"
            )
            status_lines.append(model_start_message)
            _emit_progress_line(model_start_message)
            progress((model_index - 1) / total_models, desc=model_start_message[:120])
            yield _build_progress_output(status_lines, info_blocks), preview_result
            try:
                current_file_index, current_file_index2 = _resolve_index_inputs_for_model(
                    current_model_name, loop_all_models, file_index, file_index2
                )
                if loop_all_models:
                    _load_model_for_conversion(current_model_name, protect)

                info, opt, resolved_input_path, resolved_index_path = (
                    _run_single_inference_conversion(
                        current_model_name,
                        sid,
                        input_audio_path,
                        f0_up_key,
                        f0_file,
                        f0_method,
                        current_file_index,
                        current_file_index2,
                        index_rate,
                        filter_radius,
                        resample_sr,
                        rms_mix_rate,
                        protect,
                        auto_separate_audio,
                        auto_separate_model,
                        prepared_input=prepared_input,
                    )
                )
                if opt is None:
                    info_blocks.append(f"{model_header}\n{info}")
                    continue

                target_sr, output_audio = opt
                output_base_name = _append_model_name_to_output_base(
                    base_name, current_model_name, loop_all_models
                )
                saved_sample_rate = get_supported_sample_rate_for_format(
                    audio_format, target_sr
                )
                expected_output_audio_path = output_dir / f"{output_base_name}.{audio_format}"
                expected_metadata_path = output_dir / f"{output_base_name}.txt"
                output_audio_path, metadata_path = _save_output_with_metadata(
                    output_dir,
                    output_base_name,
                    output_audio,
                    target_sr,
                    audio_format,
                    bitrate_kbps if save_as_mp3 else None,
                    _build_conversion_metadata(
                        mode="single",
                        model_name=current_model_name,
                        speaker_id=sid,
                        input_audio_path=resolved_input_path,
                        output_audio_path=expected_output_audio_path,
                        output_metadata_path=expected_metadata_path,
                        sample_rate=saved_sample_rate,
                        audio_format=audio_format,
                        f0_up_key=f0_up_key,
                        f0_method=f0_method,
                        index_path=resolved_index_path,
                        index_rate=index_rate,
                        filter_radius=filter_radius,
                        resample_sr=resample_sr,
                        rms_mix_rate=rms_mix_rate,
                        protect=protect,
                        auto_separate_audio=auto_separate_audio,
                        auto_separate_model=auto_separate_model,
                        loop_all_models=loop_all_models,
                        save_as_mp3=save_as_mp3,
                        mp3_bitrate_kbps=bitrate_kbps,
                        conversion_info=info,
                    ),
                )
                info_blocks.append(
                    f"{model_header}\n{info}\nSaved output: {output_audio_path}\nSaved metadata: {metadata_path}"
                )
                if preview_result is None or current_model_name == preview_target_model:
                    preview_result = (target_sr, output_audio)
                    preview_model = current_model_name
                model_done_message = (
                    f"[{model_index}/{total_models}] Completed model {current_model_name}"
                )
                status_lines.append(model_done_message)
                _emit_progress_line(model_done_message)
            except Exception:
                failure_info = traceback.format_exc()
                logger.warning(failure_info)
                info_blocks.append(f"{model_header}\n{failure_info}")
                model_done_message = (
                    f"[{model_index}/{total_models}] Failed model {current_model_name}"
                )
                status_lines.append(model_done_message)
                _emit_progress_line(model_done_message)
            progress(model_index / total_models, desc=model_done_message[:120])
            yield _build_progress_output(status_lines, info_blocks), preview_result

        if loop_all_models and preview_result is not None:
            info_blocks.append(f"Previewing result from model: {preview_model}")
        final_message = "Single conversion finished."
        status_lines.append(final_message)
        _emit_progress_line(final_message)
        progress(1, desc=final_message)
        yield _build_progress_output(status_lines, info_blocks), preview_result
    finally:
        _cleanup_prepared_inference_input(prepared_input)
        if loop_all_models:
            _restore_model_after_loop(model_name, protect, file_index2)


def infer_convert_batch_from_single_settings(
    model_name,
    sid,
    dir_input,
    opt_input,
    f0_up_key,
    f0_method,
    file_index,
    file_index2,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
    auto_separate_audio,
    auto_separate_model,
    loop_all_models,
    save_as_mp3,
    mp3_bitrate_kbps,
    progress=gr.Progress(track_tqdm=False),
):
    prepared_inputs = {}
    status_lines = []
    infos = []
    f0_up_key = _normalize_transpose_value(f0_up_key)
    try:
        input_dir = _resolve_input_dir(dir_input)
        output_dir = _resolve_output_dir(opt_input, require_provided=True)
        if input_dir == output_dir:
            message = "Input folder and output folder must be different."
            _emit_progress_line(message)
            yield message
            return

        audio_paths = sorted(
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in AUDIO_FILE_EXTENSIONS
        )
        if not audio_paths:
            message = f"No supported audio files found in: {input_dir}"
            _emit_progress_line(message)
            yield message
            return

        model_names = _resolve_model_names_for_conversion(model_name, loop_all_models)
        audio_format = _resolve_output_audio_format(save_as_mp3)
        bitrate_kbps = _normalize_mp3_bitrate_kbps(mp3_bitrate_kbps)
        total_jobs = max(1, len(model_names) * len(audio_paths))
        completed_jobs = 0
        start_message = (
            f"Starting batch conversion: {len(audio_paths)} file(s) x {len(model_names)} model(s) = {len(audio_paths) * len(model_names)} job(s)."
        )
        status_lines.append(start_message)
        _emit_progress_line(start_message)
        progress(0, desc=start_message[:120])
        yield _build_progress_output(status_lines, infos)
        try:
            for model_index, current_model_name in enumerate(model_names, start=1):
                model_status = (
                    f"Model {model_index}/{len(model_names)}: {current_model_name}"
                )
                status_lines.append(model_status)
                _emit_progress_line(model_status)
                yield _build_progress_output(status_lines, infos)
                try:
                    current_file_index, current_file_index2 = _resolve_index_inputs_for_model(
                        current_model_name, loop_all_models, file_index, file_index2
                    )
                    if loop_all_models:
                        _load_model_for_conversion(current_model_name, protect)

                    for input_path in audio_paths:
                        display_name = (
                            f"{input_path.name} [{current_model_name}]"
                            if loop_all_models
                            else input_path.name
                        )
                        job_number = completed_jobs + 1
                        job_start_message = (
                            f"[{job_number}/{total_jobs}] Starting {display_name}"
                        )
                        status_lines.append(job_start_message)
                        _emit_progress_line(job_start_message)
                        progress(
                            completed_jobs / total_jobs,
                            desc=job_start_message[:120],
                        )
                        yield _build_progress_output(status_lines, infos)
                        try:
                            prepared_input = prepared_inputs.get(input_path)
                            if prepared_input is None:
                                prepared_input = _prepare_inference_input(
                                    str(input_path),
                                    auto_separate_audio,
                                    auto_separate_model,
                                )
                                prepared_inputs[input_path] = prepared_input

                            info, opt, resolved_input_path, resolved_index_path = (
                                _run_single_inference_conversion(
                                    current_model_name,
                                    sid,
                                    str(input_path),
                                    f0_up_key,
                                    None,
                                    f0_method,
                                    current_file_index,
                                    current_file_index2,
                                    index_rate,
                                    filter_radius,
                                    resample_sr,
                                    rms_mix_rate,
                                    protect,
                                    auto_separate_audio,
                                    auto_separate_model,
                                    prepared_input=prepared_input,
                                )
                            )
                            if opt is None:
                                infos.append(f"{display_name} -> {info}")
                                continue

                            target_sr, output_audio = opt
                            output_base_name = _append_model_name_to_output_base(
                                input_path.stem, current_model_name, loop_all_models
                            )
                            saved_sample_rate = get_supported_sample_rate_for_format(
                                audio_format, target_sr
                            )
                            expected_output_audio_path = (
                                output_dir / f"{output_base_name}.{audio_format}"
                            )
                            expected_metadata_path = output_dir / f"{output_base_name}.txt"
                            output_audio_path, metadata_path = _save_output_with_metadata(
                                output_dir,
                                output_base_name,
                                output_audio,
                                target_sr,
                                audio_format,
                                bitrate_kbps if save_as_mp3 else None,
                                _build_conversion_metadata(
                                    mode="batch",
                                    model_name=current_model_name,
                                    speaker_id=sid,
                                    input_audio_path=resolved_input_path,
                                    output_audio_path=expected_output_audio_path,
                                    output_metadata_path=expected_metadata_path,
                                    sample_rate=saved_sample_rate,
                                    audio_format=audio_format,
                                    f0_up_key=f0_up_key,
                                    f0_method=f0_method,
                                    index_path=resolved_index_path,
                                    index_rate=index_rate,
                                    filter_radius=filter_radius,
                                    resample_sr=resample_sr,
                                    rms_mix_rate=rms_mix_rate,
                                    protect=protect,
                                    auto_separate_audio=auto_separate_audio,
                                    auto_separate_model=auto_separate_model,
                                    loop_all_models=loop_all_models,
                                    save_as_mp3=save_as_mp3,
                                    mp3_bitrate_kbps=bitrate_kbps,
                                    conversion_info=info,
                                ),
                            )
                            infos.append(
                                f"{display_name} -> Success. Saved output: {output_audio_path}. Saved metadata: {metadata_path}"
                            )
                            job_done_message = (
                                f"[{job_number}/{total_jobs}] Completed {display_name}"
                            )
                        except Exception:
                            failure_info = traceback.format_exc()
                            logger.warning(failure_info)
                            infos.append(f"{display_name} -> {failure_info}")
                            job_done_message = (
                                f"[{job_number}/{total_jobs}] Failed {display_name}"
                            )
                        completed_jobs += 1
                        status_lines.append(job_done_message)
                        _emit_progress_line(job_done_message)
                        progress(
                            completed_jobs / total_jobs,
                            desc=job_done_message[:120],
                        )
                        yield _build_progress_output(status_lines, infos)
                except Exception:
                    failure_info = traceback.format_exc()
                    logger.warning(failure_info)
                    infos.append(f"[{current_model_name}] -> {failure_info}")
                    model_failed_message = (
                        f"Failed to initialize model {current_model_name}"
                    )
                    status_lines.append(model_failed_message)
                    _emit_progress_line(model_failed_message)
                    yield _build_progress_output(status_lines, infos)
        finally:
            for prepared_input in prepared_inputs.values():
                _cleanup_prepared_inference_input(prepared_input)
            if loop_all_models:
                _restore_model_after_loop(model_name, protect, file_index2)
        final_message = "Batch conversion finished."
        status_lines.append(final_message)
        _emit_progress_line(final_message)
        progress(1, desc=final_message)
        yield _build_progress_output(status_lines, infos)
    except Exception:
        failure_info = traceback.format_exc()
        logger.warning(failure_info)
        _emit_progress_line("Batch conversion failed.")
        yield failure_info


def save_infer_preset(preset_name, selected_preset_name, *values):
    typed_name = str(preset_name or "").strip()
    selected_name = str(selected_preset_name or "").strip()
    target_name = typed_name or selected_name
    if not target_name:
        return (
            gr.update(),
            gr.update(),
            "Enter a preset name or select an existing preset to overwrite.",
        )

    payload = build_infer_preset_payload(*values)
    saved_name = INFER_PRESET_MANAGER.save_preset_safe(
        INFER_PRESET_TAB, None, target_name, payload
    )
    presets = get_infer_preset_choices()
    return (
        gr.update(choices=presets, value=saved_name),
        gr.update(value=saved_name),
        f"Saved preset '{saved_name}'.",
    )


def _build_loaded_infer_outputs(settings, preset_name, status_message):
    available_index_paths = _refresh_index_path_cache()
    preset_name = _normalize_infer_preset_name(preset_name)
    sid_value = settings["sid0"]
    speaker_value = int(settings["spk_item"] or 0)
    protect0_value = float(settings["protect0"])
    protect1_value = float(settings["protect1"])
    index_value_single = (
        settings["file_index2"] if settings["file_index2"] in available_index_paths else ""
    )
    index_value_batch = (
        settings["file_index4"] if settings["file_index4"] in available_index_paths else ""
    )
    model_info_value = ""

    spk_update = gr.update(value=speaker_value, visible=False)
    protect0_update = gr.update(value=protect0_value)
    protect1_update = gr.update(value=protect1_value, visible=False)
    file_index2_update = gr.update(
        choices=available_index_paths,
        value=index_value_single,
    )
    file_index4_update = gr.update(
        choices=available_index_paths,
        value=index_value_batch,
    )

    if sid_value:
        try:
            (
                speaker_info,
                protect0_info,
                protect1_info,
                file_index2_info,
                file_index4_info,
                model_info_value,
            ) = vc.get_vc(
                sid_value,
                protect0_value,
                protect1_value,
                index_value_single,
                index_value_batch,
            )
            speaker_max = int(speaker_info.get("maximum", speaker_value) or speaker_value)
            speaker_value = min(max(speaker_value, 0), speaker_max)
            spk_update = gr.update(
                value=speaker_value,
                visible=speaker_info.get("visible", True),
                maximum=speaker_max,
            )
            protect0_update = gr.update(
                value=protect0_value,
                visible=protect0_info.get("visible", True),
            )
            protect1_update = gr.update(value=protect1_value, visible=False)
            file_index2_value = (
                index_value_single
                if index_value_single
                else file_index2_info.get("value", "")
            )
            file_index4_value = (
                index_value_batch
                if index_value_batch
                else file_index4_info.get("value", "")
            )
            file_index2_update = gr.update(
                choices=available_index_paths,
                value=(
                    file_index2_value
                    if file_index2_value in available_index_paths or file_index2_value == ""
                    else ""
                ),
            )
            file_index4_update = gr.update(
                choices=available_index_paths,
                value=(
                    file_index4_value
                    if file_index4_value in available_index_paths or file_index4_value == ""
                    else ""
                ),
            )
        except Exception:
            logger.warning(traceback.format_exc())
            sid_value = ""
            status_message = (
                f"{status_message}\nCould not auto-load model '{settings['sid0']}'."
            )

    return (
        gr.update(choices=get_infer_preset_choices(), value=preset_name),
        gr.update(value=preset_name or ""),
        gr.update(choices=sorted(names), value=sid_value),
        spk_update,
        settings["vc_transform0"],
        file_index2_update,
        settings["f0method0"],
        settings["resample_sr0"],
        settings["rms_mix_rate0"],
        protect0_update,
        settings["filter_radius0"],
        settings["index_rate1"],
        settings["auto_separate_audio"],
        gr.update(
            choices=AUTO_SEPARATION_MODELS,
            value=settings["auto_separate_model"],
        ),
        settings["loop_all_models"],
        settings["save_as_mp3"],
        gr.update(
            value=settings["mp3_bitrate_kbps"],
            visible=bool(settings["save_as_mp3"]),
        ),
        settings["vc_transform1"],
        settings["dir_input"],
        settings["opt_input"],
        file_index4_update,
        settings["f0method1"],
        settings["resample_sr1"],
        settings["rms_mix_rate1"],
        protect1_update,
        settings["filter_radius1"],
        settings["index_rate2"],
        settings["format1"],
        model_info_value,
        status_message.strip(),
    )


def load_infer_preset(preset_name):
    if not preset_name or not str(preset_name).strip():
        settings = merge_infer_preset_with_defaults(None)
        return _build_loaded_infer_outputs(
            settings,
            None,
            "No preset selected.",
        )

    preset = INFER_PRESET_MANAGER.load_preset_safe(INFER_PRESET_TAB, None, preset_name)
    if preset is None:
        settings = merge_infer_preset_with_defaults(None)
        return _build_loaded_infer_outputs(
            settings,
            None,
            f"Preset '{preset_name}' was not found.",
        )

    INFER_PRESET_MANAGER.set_last_used(INFER_PRESET_TAB, None, preset_name)
    settings = merge_infer_preset_with_defaults(preset)
    return _build_loaded_infer_outputs(
        settings,
        preset_name,
        f"Loaded preset '{preset_name}'.",
    )


def delete_infer_preset(preset_name):
    if not preset_name or not str(preset_name).strip():
        return gr.update(), gr.update(), "No preset selected."

    deleted = INFER_PRESET_MANAGER.delete_preset(INFER_PRESET_TAB, None, preset_name)
    presets = get_infer_preset_choices()
    next_value = presets[-1] if presets else None
    if deleted:
        current_last_used = INFER_PRESET_MANAGER.get_last_used_name(INFER_PRESET_TAB, None)
        if current_last_used == preset_name:
            if next_value:
                INFER_PRESET_MANAGER.set_last_used(INFER_PRESET_TAB, None, next_value)
        return (
            gr.update(choices=presets, value=next_value),
            gr.update(value=next_value or ""),
            f"Deleted preset '{preset_name}'.",
        )
    return (
        gr.update(),
        gr.update(),
        f"Could not delete preset '{preset_name}'.",
    )


def refresh_infer_presets():
    presets = get_infer_preset_choices()
    value = _normalize_infer_preset_name(
        INFER_PRESET_MANAGER.get_last_used_name(INFER_PRESET_TAB, None)
    )
    return gr.update(choices=presets, value=value)


def restore_last_used_infer_preset():
    preset_name = _normalize_infer_preset_name(
        INFER_PRESET_MANAGER.get_last_used_name(INFER_PRESET_TAB, None)
    )
    if preset_name:
        return load_infer_preset(preset_name)
    settings = merge_infer_preset_with_defaults(None)
    return _build_loaded_infer_outputs(settings, None, "")


INITIAL_INFER_PRESET = load_initial_infer_preset()
INITIAL_INFER_PRESET_NAME = _normalize_infer_preset_name(
    INFER_PRESET_MANAGER.get_last_used_name(INFER_PRESET_TAB, None)
)
INITIAL_MODEL_INFO = ""
INITIAL_SPK_MAX = 2333
INITIAL_SPK_VISIBLE = False
INITIAL_FILE_INDEX2 = INITIAL_INFER_PRESET["file_index2"]
INITIAL_FILE_INDEX4 = INITIAL_INFER_PRESET["file_index4"]
INITIAL_PROTECT0_VISIBLE = True
INITIAL_PROTECT1_VISIBLE = False
INITIAL_MERGE_MODEL_CHOICES = _available_model_names()
INITIAL_MERGE_BASE_MODEL = (
    INITIAL_INFER_PRESET["sid0"]
    if INITIAL_INFER_PRESET["sid0"] in INITIAL_MERGE_MODEL_CHOICES
    else (INITIAL_MERGE_MODEL_CHOICES[0] if INITIAL_MERGE_MODEL_CHOICES else "")
)
(
    INITIAL_MERGE_BASE_MODEL,
    INITIAL_MERGE_COMPATIBLE_CHOICES,
    INITIAL_MERGE_COMPATIBILITY_STATUS,
    _INITIAL_MERGE_INCOMPATIBLE_LIST,
) = _scan_merge_compatible_models(INITIAL_MERGE_BASE_MODEL, INITIAL_MERGE_MODEL_CHOICES)
INITIAL_MERGE_SELECTED_MODELS = (
    []
)
INITIAL_MERGE_WEIGHT_TEXT = ""
INITIAL_MERGED_MODEL_NAME = _suggest_merged_voice_model_name(
    _build_full_merge_model_selection(
        INITIAL_MERGE_BASE_MODEL,
        INITIAL_MERGE_SELECTED_MODELS,
    ),
)

if INITIAL_INFER_PRESET["sid0"]:
    try:
        (
            _initial_speaker_info,
            _initial_protect0_info,
            _initial_protect1_info,
            _initial_file_index2_info,
            _initial_file_index4_info,
            INITIAL_MODEL_INFO,
        ) = vc.get_vc(
            INITIAL_INFER_PRESET["sid0"],
            INITIAL_INFER_PRESET["protect0"],
            INITIAL_INFER_PRESET["protect1"],
            INITIAL_INFER_PRESET["file_index2"],
            INITIAL_INFER_PRESET["file_index4"],
        )
        INITIAL_SPK_MAX = int(_initial_speaker_info.get("maximum", INITIAL_SPK_MAX) or INITIAL_SPK_MAX)
        INITIAL_SPK_VISIBLE = _initial_speaker_info.get("visible", False)
        INITIAL_PROTECT0_VISIBLE = _initial_protect0_info.get("visible", True)
        INITIAL_PROTECT1_VISIBLE = False
        INITIAL_FILE_INDEX2 = (
            INITIAL_INFER_PRESET["file_index2"]
            or _initial_file_index2_info.get("value", "")
        )
        INITIAL_FILE_INDEX4 = (
            INITIAL_INFER_PRESET["file_index4"]
            or _initial_file_index4_info.get("value", "")
        )
    except Exception:
        logger.warning(traceback.format_exc())
        INITIAL_INFER_PRESET["sid0"] = ""
        INITIAL_MODEL_INFO = ""


def clean():
    return {"value": "", "__type__": "update"}


def export_onnx(ModelPath, ExportedPath):
    from rvc.onnx import export_onnx as eo

    eo(ModelPath, ExportedPath)


sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


def if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True


def if_done_multi(done, ps):
    while 1:
        # poll==None means the process has not finished
        # Keep running as long as any process is not finished
        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    sr = sr_dict[sr]
    exp_path = pathlib.Path(now_dir, "logs", exp_dir)
    os.makedirs(exp_path, exist_ok=True)
    log_file_path = exp_path / "preprocess.log"
    f = open(log_file_path, "w")
    f.close()
    cmd = '"%s" infer/modules/train/preprocess.py "%s" %s %s "%s" %s %.1f' % (
        config.python_cmd,
        trainset_dir,
        sr,
        n_p,
        str(exp_path),
        config.noparallel,
        config.preprocess_per,
    )
    logger.info("Execute: " + cmd)
    # , stdin=PIPE, stdout=PIPE,stderr=PIPE,cwd=now_dir
    p = Popen(cmd, shell=True)
    # Gradio limitation: popen read must complete entirely before reading. Without Gradio, it reads line by line; need to create a separate text stream to read periodically
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()
    while 1:
        with open(log_file_path, "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open(log_file_path, "r") as f:
        log = f.read()
    logger.info(log)
    yield log


# but2.click(extract_f0,[gpus6,np7,f0method8,if_f0_3,trainset_dir4],[info2])
def extract_f0_feature(n_p, f0method, if_f0, exp_dir, version19):
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "w")
    f.close()
    if if_f0:
        if f0method != "rmvpe_gpu":
            cmd = (
                '"%s" infer/modules/train/extract_f0_print.py "%s/logs/%s" %s %s "%s" %s'
                % (
                    config.python_cmd,
                    now_dir,
                    exp_dir,
                    n_p,
                    f0method,
                    config.device,
                    str(config.is_half),
                )
            )
            logger.info("Execute: " + cmd)
            p = Popen(
                cmd, shell=True, cwd=now_dir
            )  # , stdin=PIPE, stdout=PIPE,stderr=PIPE
            done = [False]
            threading.Thread(
                target=if_done,
                args=(
                    done,
                    p,
                ),
            ).start()
        while 1:
            with open(
                "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r"
            ) as f:
                yield (f.read())
            sleep(1)
            if done[0]:
                break
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            log = f.read()
        logger.info(log)
        yield log
    # Start multiple processes for different parts
    """
    n_part=int(sys.argv[1])
    i_part=int(sys.argv[2])
    i_gpu=sys.argv[3]
    exp_dir=sys.argv[4]
    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)
    """
    gpu_list = [gpu for gpu in gpus.split("-") if gpu]
    if not gpu_list:
        gpu_list = ["0"] if torch.cuda.is_available() else [config.device]
    leng = len(gpu_list)
    ps = []
    for idx, n_g in enumerate(gpu_list):
        cmd = (
            '"%s" infer/modules/train/extract_feature_print.py %s %s %s %s "%s/logs/%s" %s %s'
            % (
                config.python_cmd,
                config.device,
                leng,
                idx,
                n_g,
                now_dir,
                exp_dir,
                version19,
                config.is_half,
            )
        )
        logger.info("Execute: " + cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)
    # Gradio limitation: popen read must complete entirely before reading. Without Gradio, it reads line by line; need to create a separate text stream to read periodically
    done = [False]
    threading.Thread(
        target=if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    yield log


def get_pretrained_models(path_str, f0_str, sr2):
    if_pretrained_generator_exist = os.access(
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if not if_pretrained_generator_exist:
        logger.warning(
            "assets/pretrained%s/%sG%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    if not if_pretrained_discriminator_exist:
        logger.warning(
            "assets/pretrained%s/%sD%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    return (
        (
            "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2)
            if if_pretrained_generator_exist
            else ""
        ),
        (
            "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)
            if if_pretrained_discriminator_exist
            else ""
        ),
    )


def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return get_pretrained_models(path_str, f0_str, sr2)


def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    if sr2 == "32k" and version19 == "v1":
        sr2 = "40k"
    to_return_sr2 = (
        {"choices": ["40k", "48k"], "__type__": "update", "value": sr2}
        if version19 == "v1"
        else {"choices": ["32k", "40k", "48k"], "__type__": "update", "value": sr2}
    )
    f0_str = "f0" if if_f0_3 else ""
    return (
        *get_pretrained_models(path_str, f0_str, sr2),
        to_return_sr2,
    )


def change_f0(if_f0_3, sr2, version19):  # f0method8,pretrained_G14,pretrained_D15
    path_str = "" if version19 == "v1" else "_v2"
    return (
        {"visible": if_f0_3, "__type__": "update"},
        *get_pretrained_models(path_str, "f0" if if_f0_3 == True else "", sr2),
    )


# but3.click(click_train,[exp_dir1,sr2,if_f0_3,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16])
def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
    author,
):
    # Generate filelist
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    logger.debug("Write filelist done")
    logger.info("Use gpus: %s", str(gpus16))
    if pretrained_G14 == "":
        logger.info("No pretrained Generator")
    if pretrained_D15 == "":
        logger.info("No pretrained Discriminator")
    if version19 == "v1" or sr2 == "40k":  # v2 40k falls back to v1
        config_path = "v1/%s.json" % sr2
    else:
        config_path = "v2/%s.json" % sr2
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                config.json_config[config_path],
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")
    cmd = (
        '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s -a "%s"'
        % (
            config.python_cmd,
            exp_dir1,
            sr2,
            1 if if_f0_3 else 0,
            batch_size12,
            total_epoch11,
            save_epoch10,
            '-pg "%s"' % pretrained_G14 if pretrained_G14 != "" else "",
            '-pd "%s"' % pretrained_D15 if pretrained_D15 != "" else "",
            1 if if_save_latest13 == i18n("Yes") else 0,
            1 if if_cache_gpu17 == i18n("Yes") else 0,
            1 if if_save_every_weights18 == i18n("Yes") else 0,
            version19,
            author,
        )
    )
    if gpus16:
        cmd += ' -g "%s"' % (gpus16)

    logger.info("Execute: " + cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    return "Training complete. You can check the training logs in the console or the 'train.log' file under the experiment folder."


# but4.click(train_index, [exp_dir1], info3)
def train_index(exp_dir1, version19):
    # exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    exp_dir = "logs/%s" % (exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if not os.path.exists(feature_dir):
        return "Please extract features first!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "Please extract features first!"
    infos = []
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        yield "\n".join(infos)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            logger.info(info)
            infos.append(info)
            yield "\n".join(infos)

    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    # index = faiss.index_factory(256if version19=="v1"else 768, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
    infos.append("training")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    index_save_path = "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index" % (
        exp_dir,
        n_ivf,
        index_ivf.nprobe,
        exp_dir1,
        version19,
    )
    faiss.write_index(index, index_save_path)
    infos.append(i18n("Successfully built index into") + " " + index_save_path)
    link_target = "%s/%s_IVF%s_Flat_nprobe_%s_%s_%s.index" % (
        outside_index_root,
        exp_dir1,
        n_ivf,
        index_ivf.nprobe,
        exp_dir1,
        version19,
    )
    try:
        link = os.link if platform.system() == "Windows" else os.symlink
        link(index_save_path, link_target)
        infos.append(i18n("Link index to outside folder") + " " + link_target)
    except:
        infos.append(
            i18n("Link index to outside folder")
            + " "
            + link_target
            + " "
            + i18n("Fail")
        )

    # faiss.write_index(index, '%s/added_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    # infos.append("Successfully built index, added_IVF%s_Flat_FastScan_%s.index"%(n_ivf,version19))
    yield "\n".join(infos)


# but5.click(train1key, [exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, gpus6, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17], info3)
def train1key(
    exp_dir1,
    sr2,
    if_f0_3,
    trainset_dir4,
    spk_id5,
    np7,
    f0method8,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
    author,
):
    infos = []

    def get_info_str(strr):
        infos.append(strr)
        return "\n".join(infos)

    # step1:Process data
    yield get_info_str(i18n("Step 1: Processing data"))
    [get_info_str(_) for _ in preprocess_dataset(trainset_dir4, exp_dir1, sr2, np7)]

    # step2a: Extract pitch
    yield get_info_str(i18n("step2:Pitch extraction & feature extraction"))
    [
        get_info_str(_)
        for _ in extract_f0_feature(
            np7,
            f0method8,
            if_f0_3,
            exp_dir1,
            version19,
        )
    ]

    # step3a:Train model
    yield get_info_str(i18n("Step 3a: Model training started"))
    click_train(
        exp_dir1,
        sr2,
        if_f0_3,
        spk_id5,
        save_epoch10,
        total_epoch11,
        batch_size12,
        if_save_latest13,
        pretrained_G14,
        pretrained_D15,
        gpus16,
        if_cache_gpu17,
        if_save_every_weights18,
        version19,
        author,
    )
    yield get_info_str(
        i18n(
            "Training complete. You can check the training logs in the console or the 'train.log' file under the experiment folder."
        )
    )

    # step3b: Train index
    [get_info_str(_) for _ in train_index(exp_dir1, version19)]
    yield get_info_str(i18n("All processes have been completed!"))


#                    ckpt_path2.change(change_info_,[ckpt_path2],[sr__,if_f0__])
def change_info_(ckpt_path):
    if not os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path), "train.log")):
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        with open(
            ckpt_path.replace(os.path.basename(ckpt_path), "train.log"), "r"
        ) as f:
            info = eval(f.read().strip("\n").split("\n")[0].split("\t")[-1])
            sr, f0 = info["sample_rate"], info["if_f0"]
            version = "v2" if ("version" in info and info["version"] == "v2") else "v1"
            return sr, str(f0), version
    except:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}

theme = gr.themes.Ocean(
    primary_hue="emerald",
    secondary_hue="sky",
    neutral_hue="slate",
    spacing_size=gr.themes.sizes.spacing_md,
    radius_size=gr.themes.sizes.radius_lg,
    text_size=gr.themes.sizes.text_md,
    font=(
        gr.themes.GoogleFont("Sora"),
        "ui-sans-serif",
        "sans-serif",
    ),
    font_mono=(
        gr.themes.GoogleFont("JetBrains Mono"),
        "ui-monospace",
        "monospace",
    ),
).set(
    body_background_fill="linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%)",
    body_background_fill_dark="linear-gradient(180deg, #0b1220 0%, #0f172a 100%)",
    background_fill_primary="rgba(248, 250, 252, 0.92)",
    background_fill_primary_dark="rgba(15, 23, 42, 0.94)",
    block_background_fill="rgba(255, 255, 255, 0.85)",
    block_background_fill_dark="rgba(15, 23, 42, 0.75)",
    block_border_color="rgba(148, 163, 184, 0.45)",
    block_border_color_dark="rgba(51, 65, 85, 0.8)",
    block_border_width="1px",
    block_shadow="0 18px 40px -32px rgba(15, 23, 42, 0.6)",
    panel_background_fill="rgba(255, 255, 255, 0.6)",
    panel_background_fill_dark="rgba(2, 6, 23, 0.6)",
    input_background_fill="rgba(255, 255, 255, 0.95)",
    input_background_fill_dark="rgba(15, 23, 42, 0.85)",
    input_border_color="rgba(148, 163, 184, 0.6)",
    input_border_color_dark="rgba(71, 85, 105, 0.9)",
    input_border_width="1px",
    button_primary_text_color="white",
)

MERGE_VOICE_CSS = """
#merge-voice-accordion {
    border-radius: 24px;
    overflow: visible;
}

#merge-voice-accordion .label-wrap {
    align-items: center;
    padding: 16px 18px;
    border: none;
    border-radius: 22px;
    background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 45%, #e11d48 100%);
    color: #fff;
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.18),
        0 18px 34px -24px rgba(127, 29, 29, 0.88);
    transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
}

#merge-voice-accordion .label-wrap:hover {
    transform: translateY(-1px);
    filter: saturate(1.05);
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.18),
        0 22px 38px -24px rgba(127, 29, 29, 0.95);
}

#merge-voice-accordion .label-wrap span,
#merge-voice-accordion .label-wrap .icon {
    color: inherit;
}

#merge-voice-accordion > div:last-child {
    padding-top: 8px;
}

#merge-voice-accordion .merge-voice-action.primary {
    min-height: 50px;
    border: 1px solid rgba(127, 29, 29, 0.34);
    border-radius: 999px;
    background: linear-gradient(135deg, #991b1b 0%, #dc2626 48%, #fb7185 100%);
    color: #fff;
    box-shadow: 0 22px 34px -24px rgba(190, 24, 93, 0.9);
    font-weight: 700;
    letter-spacing: 0.01em;
    transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
}

#merge-voice-accordion .merge-voice-action.primary:hover {
    transform: translateY(-1px);
    filter: brightness(1.03);
    box-shadow: 0 28px 42px -24px rgba(190, 24, 93, 1);
}

#merge-voice-accordion .merge-voice-action.primary:focus-visible {
    box-shadow:
        0 0 0 3px rgba(251, 113, 133, 0.25),
        0 28px 42px -24px rgba(190, 24, 93, 1);
}

body.dark #merge-voice-accordion,
.dark #merge-voice-accordion {
    background: transparent;
    box-shadow: none;
}
"""

with gr.Blocks(title="RVC WebUI", theme=theme, css=MERGE_VOICE_CSS) as app:
    gr.Markdown("## SECourses Premium RVC WebUI V5.1 : https://www.patreon.com/posts/149104996")
    gr.Markdown("### Pre-Trained Voices Download 1 : https://huggingface.co/QuickWick/Music-AI-Voices/tree/main")
    gr.Markdown("### Pre-Trained Voices Download 2 : https://huggingface.co/Coolwowsocoolwow")
    gr.Markdown(
        value=i18n(
            "This is a software based on open source models. Users who use the software and distribute the sounds exported by the software are solely responsible. Use responsibly."
        )
    )
    with gr.Tabs():
        with gr.TabItem(i18n("Model Inference")):
            with gr.Row():
                sid0 = gr.Dropdown(
                    label=i18n("Inferencing voice"),
                    choices=sorted(names),
                    value=INITIAL_INFER_PRESET["sid0"],
                )
                with gr.Column():
                    refresh_button = gr.Button(
                        i18n("Refresh voice list and index path"), variant="primary"
                    )
                    clean_button = gr.Button(
                        i18n("Unload model to save GPU memory"), variant="primary"
                    )
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=INITIAL_SPK_MAX,
                    step=1,
                    label=i18n("Select Speaker/Singer ID"),
                    value=min(int(INITIAL_INFER_PRESET["spk_item"] or 0), INITIAL_SPK_MAX),
                    visible=INITIAL_SPK_VISIBLE,
                    interactive=True,
                )
                clean_button.click(
                    fn=clean, inputs=[], outputs=[sid0], api_name="infer_clean"
                )
            loop_all_models = gr.Checkbox(
                label=i18n("Loop all models"),
                value=INITIAL_INFER_PRESET["loop_all_models"],
                interactive=True,
                info=i18n(
                    "Run the same input through every installed voice model and append the model name to each saved output filename. The in-app preview will play one generated result."
                ),
            )
            infer_preset_choices = get_infer_preset_choices()
            with gr.Row():
                with gr.Column(scale=3):
                    modelinfo = gr.Textbox(
                        label=i18n("Model info"),
                        max_lines=8,
                        value=INITIAL_MODEL_INFO,
                    )
                    open_outputs_button = gr.Button(
                        i18n("Open outputs folder")
                    )
                with gr.Column(scale=2):
                    auto_separate_audio = gr.Checkbox(
                        label=i18n(
                            "Auto separate vocals + music and remix after conversion"
                        ),
                        value=INITIAL_INFER_PRESET["auto_separate_audio"],
                        interactive=True,
                    )
                    auto_separate_model = gr.Dropdown(
                        label=i18n("Auto separation model"),
                        choices=AUTO_SEPARATION_MODELS,
                        value=INITIAL_INFER_PRESET["auto_separate_model"],
                        interactive=True,
                    )
                    with gr.Row():
                        save_as_mp3 = gr.Checkbox(
                            label=i18n("Save as MP3"),
                            value=INITIAL_INFER_PRESET["save_as_mp3"],
                            interactive=True,
                            info=i18n(
                                "When enabled, saved outputs are written as MP3 instead of WAV for both single and batch conversion."
                            ),
                        )
                        mp3_bitrate_kbps = gr.Slider(
                            minimum=32,
                            maximum=320,
                            step=8,
                            label=i18n("MP3 bitrate (kbps)"),
                            value=INITIAL_INFER_PRESET["mp3_bitrate_kbps"],
                            interactive=True,
                            visible=bool(INITIAL_INFER_PRESET["save_as_mp3"]),
                            info=i18n(
                                "Used only when Save as MP3 is enabled. Higher bitrates preserve more detail but increase file size."
                            ),
                        )
                with gr.Column(scale=3):
                    infer_preset_dropdown = gr.Dropdown(
                        label=i18n("Inference preset"),
                        choices=infer_preset_choices,
                        value=INITIAL_INFER_PRESET_NAME,
                        allow_custom_value=False,
                        interactive=True,
                    )
                    infer_preset_name = gr.Textbox(
                        label=i18n("Preset name"),
                        value=INITIAL_INFER_PRESET_NAME or "",
                    )
                    with gr.Row():
                        infer_preset_refresh = gr.Button(i18n("Refresh"))
                        infer_preset_save = gr.Button(i18n("Save"))
                        infer_preset_load = gr.Button(i18n("Load"))
                        infer_preset_delete = gr.Button(i18n("Delete"))
                    infer_preset_status = gr.Textbox(
                        label=i18n("Preset status"),
                        value="",
                    )
            with gr.Accordion(
                i18n("Make Merged Voice"),
                open=False,
                elem_id="merge-voice-accordion",
            ):
                gr.Markdown(
                    value=i18n(
                        "Pick two or more installed voice models to create a weighted merged voice in the weights folder. Leave weights blank for an equal mix. The new voice is loaded automatically after it is created."
                    )
                )
                with gr.Row():
                    merge_base_model = gr.Dropdown(
                        label=i18n("Compatibility base voice"),
                        choices=INITIAL_MERGE_MODEL_CHOICES,
                        value=INITIAL_MERGE_BASE_MODEL,
                        interactive=True,
                        info=i18n(
                            "Selecting a base voice scans all installed models and filters this list to compatible merge candidates."
                        ),
                    )
                with gr.Row():
                    merge_model_names = gr.Dropdown(
                        label=i18n("Additional compatible voices to merge"),
                        choices=[
                            model_name
                            for model_name in INITIAL_MERGE_COMPATIBLE_CHOICES
                            if model_name != INITIAL_MERGE_BASE_MODEL
                        ],
                        value=INITIAL_MERGE_SELECTED_MODELS,
                        multiselect=True,
                        interactive=True,
                        info=i18n(
                            "The base voice is always included automatically. This list only shows additional compatible voices."
                        ),
                    )
                with gr.Row():
                    merge_model_weights = gr.Textbox(
                        label=i18n("Weights (optional, comma-separated)"),
                        value=INITIAL_MERGE_WEIGHT_TEXT,
                        interactive=True,
                        info=i18n(
                            "Match the selected voice order. Example: 0.5, 0.3, 0.2 or 50%, 30%, 20%."
                        ),
                    )
                    merged_model_name = gr.Textbox(
                        label=i18n("Merged voice name (without extension)"),
                        value=INITIAL_MERGED_MODEL_NAME,
                        interactive=True,
                    )
                with gr.Row():
                    gr.Markdown(
                        value=i18n(
                            "Tip: repeated merges are supported, so you can still build larger blends by merging a previous merged voice with more voices later."
                        )
                    )
                merge_compatibility_status = gr.Textbox(
                    label=i18n("Compatibility scan status"),
                    value=INITIAL_MERGE_COMPATIBILITY_STATUS,
                    max_lines=12,
                    interactive=False,
                )
                merged_model_info = gr.Textbox(
                    label=i18n("Merged voice notes (optional)"),
                    value="",
                    max_lines=4,
                    interactive=True,
                )
                merge_source_indices = gr.Checkbox(
                    label=i18n("Also create merged index"),
                    value=True,
                    interactive=True,
                    info=i18n(
                        "Builds a new FAISS index from both source voice indexes when they are available."
                    ),
                )
                merge_voice_button = gr.Button(
                    i18n("Compose Merged Voice"),
                    variant="primary",
                    elem_classes=["merge-voice-action"],
                )
                merge_voice_status = gr.Textbox(
                    label=i18n("Merged voice status"),
                    value="",
                )
            with gr.TabItem(i18n("Single inference")):
                with gr.Row():
                    with gr.Column():
                        vc_transform0 = gr.Slider(
                            minimum=TRANSPOSE_MIN,
                            maximum=TRANSPOSE_MAX,
                            step=1,
                            label=i18n(
                                "Transpose (integer, number of semitones, raise by an octave: 12, lower by an octave: -12)"
                            ),
                            info=i18n(
                                "Shifts the input vocal pitch before conversion. Use 0 to keep the original pitch, positive values to make it higher, and negative values to make it lower. This changes musical key and vocal range, not the target voice identity."
                            ),
                            value=INITIAL_INFER_PRESET["vc_transform0"],
                            interactive=True,
                        )
                        input_audio0 = gr.Audio(
                            label=i18n("The audio file to be processed"),
                            type="filepath",
                        )
                        file_index2 = gr.Dropdown(
                            label=i18n(
                                "Auto-detect index path and select from the dropdown"
                            ),
                            choices=sorted(index_paths),
                            value=INITIAL_FILE_INDEX2 if INITIAL_FILE_INDEX2 in index_paths else "",
                            interactive=True,
                        )
                        file_index1 = gr.File(
                            label=i18n(
                                "Path to the feature index file. Leave blank to use the selected result from the dropdown"
                            ),
                        )
                    with gr.Column():
                        f0method0 = gr.Radio(
                            label=i18n(
                                "Select the pitch extraction algorithm ('pm': faster extraction but lower-quality speech; 'harvest': better bass but extremely slow; 'crepe': better quality but GPU intensive), 'rmvpe': RECOMMENDED - best quality, and little GPU requirement"
                            ),
                            choices=(
                                ["pm", "dio", "harvest", "crepe", "rmvpe", "fcpe"]
                            ),
                            value=INITIAL_INFER_PRESET["f0method0"],
                            interactive=True,
                        )
                        resample_sr0 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=i18n(
                                "Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling"
                            ),
                            value=INITIAL_INFER_PRESET["resample_sr0"],
                            step=1,
                            interactive=True,
                        )
                        rms_mix_rate0 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n(
                                "Adjust the volume envelope scaling. Closer to 0, the more it mimicks the volume of the original vocals. Can help mask noise and make volume sound more natural when set relatively low. Closer to 1 will be more of a consistently loud volume. Recommended: 0.25 for natural sound"
                            ),
                            value=INITIAL_INFER_PRESET["rms_mix_rate0"],
                            interactive=True,
                        )
                        protect0 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n(
                                "Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy. Recommended: 0.33"
                            ),
                            value=INITIAL_INFER_PRESET["protect0"],
                            step=0.01,
                            visible=INITIAL_PROTECT0_VISIBLE,
                            interactive=True,
                        )
                        filter_radius0 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=i18n(
                                "If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness. Recommended: 3"
                            ),
                            value=INITIAL_INFER_PRESET["filter_radius0"],
                            step=1,
                            interactive=True,
                        )
                        index_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("Feature searching ratio (higher = better timbre accuracy, recommended: 0.88-1.0)"),
                            value=INITIAL_INFER_PRESET["index_rate1"],
                            interactive=True,
                        )
                        f0_file = gr.File(
                            label=i18n(
                                "F0 curve file (optional). One pitch per line. Replaces the default F0 and pitch modulation"
                            ),
                            visible=False,
                        )
                        but0 = gr.Button(i18n("Convert"), variant="primary")
                        vc_output2 = gr.Audio(
                            label=i18n(
                                "Export audio (click on the three dots in the lower right corner to download)"
                            )
                        )

                        refresh_button.click(
                            fn=refresh_infer_model_and_index_choices,
                            inputs=[sid0, file_index2],
                            outputs=[sid0, file_index2],
                            api_name="infer_refresh",
                        )

                vc_output1 = gr.Textbox(label=i18n("Output information"))

                but0.click(
                    infer_convert_single,
                    [
                        sid0,
                        spk_item,
                        input_audio0,
                        vc_transform0,
                        f0_file,
                        f0method0,
                        file_index1,
                        file_index2,
                        # file_big_npy1,
                        index_rate1,
                        filter_radius0,
                        resample_sr0,
                        rms_mix_rate0,
                        protect0,
                        auto_separate_audio,
                        auto_separate_model,
                        loop_all_models,
                        save_as_mp3,
                        mp3_bitrate_kbps,
                    ],
                    [vc_output1, vc_output2],
                    api_name="infer_convert",
                )
            with gr.TabItem(i18n("Batch inference")):
                gr.Markdown(
                    value=i18n(
                        "Batch folder conversion uses the current Single inference model, index, and settings. Input and output folders are required. Global controls above also apply here, including auto split/remix, looping all models, and MP3 saving."
                    )
                )
                with gr.Row():
                    with gr.Column():
                        vc_transform1 = gr.Slider(
                            minimum=TRANSPOSE_MIN,
                            maximum=TRANSPOSE_MAX,
                            step=1,
                            label=i18n(
                                "Transpose (integer, number of semitones, raise by an octave: 12, lower by an octave: -12)"
                            ),
                            value=INITIAL_INFER_PRESET["vc_transform1"],
                            interactive=True,
                            visible=False,
                        )
                        dir_input = gr.Textbox(
                            label=i18n(
                                "Enter the path of the audio folder to be processed (copy it from the address bar of the file manager)"
                            ),
                            value=INITIAL_INFER_PRESET["dir_input"],
                            placeholder="C:\\Users\\Desktop\\input_vocal_dir",
                        )
                        inputs = gr.File(
                            file_count="multiple",
                            label=i18n(
                                "Multiple audio files can also be imported. If a folder path exists, this input is ignored."
                            ),
                            visible=False,
                        )
                        opt_input = gr.Textbox(
                            label=i18n("Specify output folder"),
                            value=INITIAL_INFER_PRESET["opt_input"],
                            placeholder="C:\\Users\\Desktop\\output_vocal_dir",
                        )
                        file_index4 = gr.Dropdown(
                            label=i18n(
                                "Auto-detect index path and select from the dropdown"
                            ),
                            choices=sorted(index_paths),
                            value=INITIAL_FILE_INDEX4 if INITIAL_FILE_INDEX4 in index_paths else "",
                            interactive=True,
                            visible=False,
                        )
                        file_index3 = gr.File(
                            label=i18n(
                                "Path to the feature index file. Leave blank to use the selected result from the dropdown"
                            ),
                            visible=False,
                        )

                        refresh_button.click(
                            fn=refresh_batch_index_choices,
                            inputs=[file_index4],
                            outputs=file_index4,
                            api_name="infer_refresh_batch",
                        )
                        # file_big_npy2 = gr.Textbox(
                        #     label=i18n("Feature file path"),
                        #     value="E:\\codes\\py39\\vits_vc_gpu_train\\logs\\mi-test-1key\\total_fea.npy",
                        #     interactive=True,
                        # )

                    with gr.Column():
                        f0method1 = gr.Radio(
                            label=i18n(
                                "Select the pitch extraction algorithm ('pm': faster extraction but lower-quality speech; 'harvest': better bass but extremely slow; 'crepe': better quality but GPU intensive), 'rmvpe': RECOMMENDED - best quality, and little GPU requirement"
                            ),
                            choices=(
                                ["pm", "dio", "harvest", "crepe", "rmvpe", "fcpe"]
                            ),
                            value=INITIAL_INFER_PRESET["f0method1"],
                            interactive=True,
                            visible=False,
                        )
                        resample_sr1 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=i18n(
                                "Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling"
                            ),
                            value=INITIAL_INFER_PRESET["resample_sr1"],
                            step=1,
                            interactive=True,
                            visible=False,
                        )
                        rms_mix_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n(
                                "Adjust the volume envelope scaling. Closer to 0, the more it mimicks the volume of the original vocals. Can help mask noise and make volume sound more natural when set relatively low. Closer to 1 will be more of a consistently loud volume. Recommended: 0.25 for natural sound"
                            ),
                            value=INITIAL_INFER_PRESET["rms_mix_rate1"],
                            interactive=True,
                            visible=False,
                        )
                        protect1 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n(
                                "Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy. Recommended: 0.33"
                            ),
                            value=INITIAL_INFER_PRESET["protect1"],
                            step=0.01,
                            visible=False,
                            interactive=True,
                        )
                        filter_radius1 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=i18n(
                                "If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness. Recommended: 3"
                            ),
                            value=INITIAL_INFER_PRESET["filter_radius1"],
                            step=1,
                            interactive=True,
                            visible=False,
                        )
                        index_rate2 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("Feature searching ratio (higher = better timbre accuracy, recommended: 0.88-1.0)"),
                            value=INITIAL_INFER_PRESET["index_rate2"],
                            interactive=True,
                            visible=False,
                        )
                        format1 = gr.Radio(
                            label=i18n("Export file format"),
                            choices=["wav", "flac", "mp3", "m4a"],
                            value=INITIAL_INFER_PRESET["format1"],
                            interactive=True,
                            visible=False,
                        )
                        but1 = gr.Button(i18n("Convert"), variant="primary")
                        vc_output3 = gr.Textbox(label=i18n("Output information"))

                but1.click(
                    infer_convert_batch_from_single_settings,
                    [
                        sid0,
                        spk_item,
                        dir_input,
                        opt_input,
                        vc_transform0,
                        f0method0,
                        file_index1,
                        file_index2,
                        index_rate1,
                        filter_radius0,
                        resample_sr0,
                        rms_mix_rate0,
                        protect0,
                        auto_separate_audio,
                        auto_separate_model,
                        loop_all_models,
                        save_as_mp3,
                        mp3_bitrate_kbps,
                    ],
                    [vc_output3],
                    api_name="infer_convert_batch",
                )
                save_as_mp3.change(
                    fn=toggle_mp3_bitrate_visibility,
                    inputs=[save_as_mp3],
                    outputs=[mp3_bitrate_kbps],
                )
                open_outputs_button.click(
                    fn=open_outputs_folder,
                    inputs=[],
                    outputs=[],
                )
                merge_base_model.change(
                    fn=stream_scan_merge_compatibility,
                    inputs=[merge_base_model, merge_model_names],
                    outputs=[
                        merge_base_model,
                        merge_model_names,
                        merged_model_name,
                        merge_compatibility_status,
                    ],
                    api_name="infer_scan_merge_compatibility",
                    show_progress="hidden",
                    trigger_mode="always_last",
                )
                merge_model_names.change(
                    fn=suggest_merged_voice_model_name_from_merge_inputs,
                    inputs=[merge_base_model, merge_model_names],
                    outputs=[merged_model_name],
                    show_progress="hidden",
                    queue=False,
                    trigger_mode="always_last",
                )
                refresh_button.click(
                    fn=stream_refresh_merge_compatibility_inputs,
                    inputs=[merge_base_model, merge_model_names],
                    outputs=[
                        merge_base_model,
                        merge_model_names,
                        merged_model_name,
                        merge_compatibility_status,
                    ],
                    api_name="infer_refresh_merge_voices",
                    show_progress="hidden",
                )
                merge_voice_button.click(
                    fn=create_merged_voice_model,
                    inputs=[
                        merge_base_model,
                        merge_model_names,
                        merge_model_weights,
                        merged_model_name,
                        merged_model_info,
                        merge_source_indices,
                        protect0,
                        protect1,
                        file_index2,
                        file_index4,
                    ],
                    outputs=[
                        merge_voice_status,
                        sid0,
                        merge_base_model,
                        merge_model_names,
                        merge_compatibility_status,
                        spk_item,
                        protect0,
                        protect1,
                        file_index2,
                        file_index4,
                        modelinfo,
                    ],
                    api_name="infer_merge_voice",
                )
                sid0.change(
                    fn=get_vc_for_infer_ui,
                    inputs=[sid0, protect0, protect1, file_index2, file_index4],
                    outputs=[
                        spk_item,
                        protect0,
                        protect1,
                        file_index2,
                        file_index4,
                        modelinfo,
                    ],
                    api_name="infer_change_voice",
                )
                infer_preset_refresh.click(
                    fn=refresh_infer_presets,
                    inputs=[],
                    outputs=[infer_preset_dropdown],
                )
                infer_preset_save.click(
                    fn=save_infer_preset,
                    inputs=[
                        infer_preset_name,
                        infer_preset_dropdown,
                        sid0,
                        spk_item,
                        vc_transform0,
                        file_index2,
                        f0method0,
                        resample_sr0,
                        rms_mix_rate0,
                        protect0,
                        filter_radius0,
                        index_rate1,
                        auto_separate_audio,
                        auto_separate_model,
                        loop_all_models,
                        save_as_mp3,
                        mp3_bitrate_kbps,
                        vc_transform1,
                        dir_input,
                        opt_input,
                        file_index4,
                        f0method1,
                        resample_sr1,
                        rms_mix_rate1,
                        protect1,
                        filter_radius1,
                        index_rate2,
                        format1,
                    ],
                    outputs=[
                        infer_preset_dropdown,
                        infer_preset_name,
                        infer_preset_status,
                    ],
                )
                infer_preset_delete.click(
                    fn=delete_infer_preset,
                    inputs=[infer_preset_dropdown],
                    outputs=[
                        infer_preset_dropdown,
                        infer_preset_name,
                        infer_preset_status,
                    ],
                )
                infer_preset_outputs = [
                    infer_preset_dropdown,
                    infer_preset_name,
                    sid0,
                    spk_item,
                    vc_transform0,
                    file_index2,
                    f0method0,
                    resample_sr0,
                    rms_mix_rate0,
                    protect0,
                    filter_radius0,
                    index_rate1,
                    auto_separate_audio,
                    auto_separate_model,
                    loop_all_models,
                    save_as_mp3,
                    mp3_bitrate_kbps,
                    vc_transform1,
                    dir_input,
                    opt_input,
                    file_index4,
                    f0method1,
                    resample_sr1,
                    rms_mix_rate1,
                    protect1,
                    filter_radius1,
                    index_rate2,
                    format1,
                    modelinfo,
                    infer_preset_status,
                ]
                infer_preset_dropdown.change(
                    fn=load_infer_preset,
                    inputs=[infer_preset_dropdown],
                    outputs=infer_preset_outputs,
                )
                infer_preset_load.click(
                    fn=load_infer_preset,
                    inputs=[infer_preset_dropdown],
                    outputs=infer_preset_outputs,
                )
                app.load(
                    fn=restore_last_used_infer_preset,
                    inputs=[],
                    outputs=infer_preset_outputs,
                )
        with gr.TabItem(
            i18n("Vocals/Accompaniment Separation & Reverberation Removal")
        ):
            gr.Markdown(
                value=i18n(
                    "Batch processing for vocal accompaniment separation using the UVR5 model.<br>Example of a valid folder path format: D:\\path\\to\\input\\folder (copy it from the file manager address bar).<br>The model is divided into three categories:<br>1. Preserve vocals: Choose this option for audio without harmonies. It preserves vocals better than HP5. It includes two built-in models: HP2 and HP3. HP3 may slightly leak accompaniment but preserves vocals slightly better than HP2.<br>2. Preserve main vocals only: Choose this option for audio with harmonies. It may weaken the main vocals. It includes one built-in model: HP5.<br>3. De-reverb and de-delay models (by FoxJoy):<br>  (1) MDX-Net: The best choice for stereo reverb removal but cannot remove mono reverb;<br>&emsp;(234) DeEcho: Removes delay effects. Aggressive mode removes more thoroughly than Normal mode. DeReverb additionally removes reverb and can remove mono reverb, but not very effectively for heavily reverberated high-frequency content.<br>De-reverb/de-delay notes:<br>1. The processing time for the DeEcho-DeReverb model is approximately twice as long as the other two DeEcho models.<br>2. The MDX-Net-Dereverb model is quite slow.<br>3. The recommended cleanest configuration is to apply MDX-Net first and then DeEcho-Aggressive."
                )
            )
            with gr.Row():
                with gr.Column():
                    dir_wav_input = gr.Textbox(
                        label=i18n(
                            "Enter the path of the audio folder to be processed"
                        ),
                        placeholder="C:\\Users\\Desktop\\todo-songs",
                    )
                    wav_inputs = gr.File(
                        file_count="multiple",
                        label=i18n(
                            "Multiple audio files can also be imported. If a folder path exists, this input is ignored."
                        ),
                    )
                with gr.Column():
                    model_choose = gr.Dropdown(label=i18n("Model"), choices=uvr5_names)
                    agg = gr.Slider(
                        minimum=0,
                        maximum=20,
                        step=1,
                        label="Vocal extraction aggressiveness",
                        value=10,
                        interactive=True,
                        visible=False,  # Not open for adjustment yet
                    )
                    opt_vocal_root = gr.Textbox(
                        label=i18n("Specify the output folder for vocals"),
                        value="opt",
                    )
                    opt_ins_root = gr.Textbox(
                        label=i18n("Specify the output folder for accompaniment"),
                        value="opt",
                    )
                    format0 = gr.Radio(
                        label=i18n("Export file format"),
                        choices=["wav", "flac", "mp3", "m4a"],
                        value="flac",
                        interactive=True,
                    )
                but2 = gr.Button(i18n("Convert"), variant="primary")
                vc_output4 = gr.Textbox(label=i18n("Output information"))
                but2.click(
                    uvr,
                    [
                        model_choose,
                        dir_wav_input,
                        opt_vocal_root,
                        wav_inputs,
                        opt_ins_root,
                        agg,
                        format0,
                    ],
                    [vc_output4],
                    api_name="uvr_convert",
                )
        with gr.TabItem(i18n("Train")):
            gr.Markdown(
                value=i18n(
                    "### Step 1. Fill in the experimental configuration.\nExperimental data is stored in the 'logs' folder, with each experiment having a separate folder. Manually enter the experiment name path, which contains the experimental configuration, logs, and trained model files."
                )
            )
            with gr.Row():
                exp_dir1 = gr.Textbox(
                    label=i18n("Enter the experiment name"), value="mi-test"
                )
                author = gr.Textbox(label=i18n("Model Author (Nullable)"))
                np7 = gr.Slider(
                    minimum=0,
                    maximum=config.n_cpu,
                    step=1,
                    label=i18n(
                        "Number of CPU processes used for pitch extraction and data processing"
                    ),
                    value=int(np.ceil(config.n_cpu / 1.5)),
                    interactive=True,
                )
            with gr.Row():
                sr2 = gr.Radio(
                    label=i18n("Target sample rate"),
                    choices=["32k", "40k", "48k"],
                    value="48k",
                    interactive=True,
                )
                if_f0_3 = gr.Radio(
                    label=i18n(
                        "Whether the model has pitch guidance (required for singing, optional for speech)"
                    ),
                    choices=[i18n("Yes"), i18n("No")],
                    value=i18n("Yes"),
                    interactive=True,
                )
                version19 = gr.Radio(
                    label=i18n("Version"),
                    choices=["v1", "v2"],
                    value="v2",
                    interactive=True,
                    visible=True,
                )
            gr.Markdown(
                value=i18n(
                    "### Step 2. Audio processing. \n#### 1. Slicing.\nAutomatically traverse all files in the training folder that can be decoded into audio and perform slice normalization. Generates 2 wav folders in the experiment directory. Currently, only single-singer/speaker training is supported."
                )
            )
            with gr.Row():
                with gr.Column():
                    trainset_dir4 = gr.Textbox(
                        label=i18n("Enter the path of the training folder"),
                    )
                    spk_id5 = gr.Slider(
                        minimum=0,
                        maximum=4,
                        step=1,
                        label=i18n("Please specify the speaker/singer ID"),
                        value=0,
                        interactive=True,
                    )
                    but1 = gr.Button(i18n("Process data"), variant="primary")
                with gr.Column():
                    info1 = gr.Textbox(label=i18n("Output information"), value="")
                    but1.click(
                        preprocess_dataset,
                        [trainset_dir4, exp_dir1, sr2, np7],
                        [info1],
                        api_name="train_preprocess",
                    )
            gr.Markdown(
                value=i18n(
                    "#### 2. Feature extraction.\nUse CPU to extract pitch (if the model has pitch), use GPU to extract features (select GPU index)."
                )
            )
            with gr.Row():
                with gr.Column():
                    gpu_info9 = gr.Textbox(
                        label=i18n("GPU Information"),
                        value=gpu_info,
                    )
                    f0method8 = gr.Radio(
                        label=i18n(
                            "Select the pitch extraction algorithm: when extracting singing, you can use 'pm' to speed up. For high-quality speech with fast performance, but worse CPU usage, you can use 'dio'. 'harvest' results in better quality but is slower.  'rmvpe' has the best results and consumes less CPU/GPU"
                        ),
                        choices=["pm", "dio", "harvest", "crepe", "rmvpe", "fcpe"],
                        value="rmvpe",
                        interactive=True,
                    )
                with gr.Column():
                    but2 = gr.Button(i18n("Feature extraction"), variant="primary")
                    info2 = gr.Textbox(label=i18n("Output information"), value="")
                but2.click(
                    extract_f0_feature,
                    [
                        np7,
                        f0method8,
                        if_f0_3,
                        exp_dir1,
                        version19,
                    ],
                    [info2],
                    api_name="train_extract_f0_feature",
                )
            gr.Markdown(
                value=i18n(
                    "### Step 3. Start training.\nFill in the training settings and start training the model and index."
                )
            )
            with gr.Row():
                with gr.Column():
                    save_epoch10 = gr.Slider(
                        minimum=1,
                        maximum=50,
                        step=1,
                        label=i18n("Save frequency (save_every_epoch)"),
                        value=5,
                        interactive=True,
                    )
                    total_epoch11 = gr.Slider(
                        minimum=2,
                        maximum=1000,
                        step=1,
                        label=i18n("Total training epochs (total_epoch)"),
                        value=20,
                        interactive=True,
                    )
                    batch_size12 = gr.Slider(
                        minimum=1,
                        maximum=40,
                        step=1,
                        label=i18n("Batch size per GPU"),
                        value=default_batch_size,
                        interactive=True,
                    )
                    if_save_latest13 = gr.Radio(
                        label=i18n(
                            "Save only the latest '.ckpt' file to save disk space"
                        ),
                        choices=[i18n("Yes"), i18n("No")],
                        value=i18n("No"),
                        interactive=True,
                    )
                    if_cache_gpu17 = gr.Radio(
                        label=i18n(
                            "Cache all training sets to GPU memory. Caching small datasets (less than 10 minutes) can speed up training, but caching large datasets will consume a lot of GPU memory and may not provide much speed improvement"
                        ),
                        choices=[i18n("Yes"), i18n("No")],
                        value=i18n("No"),
                        interactive=True,
                    )
                    if_save_every_weights18 = gr.Radio(
                        label=i18n(
                            "Save a small final model to the 'weights' folder at each save point"
                        ),
                        choices=[i18n("Yes"), i18n("No")],
                        value=i18n("No"),
                        interactive=True,
                    )
                with gr.Column():
                    pretrained_G14 = gr.Textbox(
                        label=i18n("Load pre-trained base model G path"),
                        value="assets/pretrained_v2/f0G48k.pth",
                        interactive=True,
                    )
                    pretrained_D15 = gr.Textbox(
                        label=i18n("Load pre-trained base model D path"),
                        value="assets/pretrained_v2/f0D48k.pth",
                        interactive=True,
                    )
                    gpus16 = gr.Textbox(
                        label=i18n(
                            "Enter the GPU index(es) separated by '-', e.g., 0-1-2 to use GPU 0, 1, and 2"
                        ),
                        value=gpus,
                        interactive=True,
                    )
                    sr2.change(
                        change_sr2,
                        [sr2, if_f0_3, version19],
                        [pretrained_G14, pretrained_D15],
                    )
                    version19.change(
                        change_version19,
                        [sr2, if_f0_3, version19],
                        [pretrained_G14, pretrained_D15, sr2],
                    )
                    if_f0_3.change(
                        change_f0,
                        [if_f0_3, sr2, version19],
                        [f0method8, pretrained_G14, pretrained_D15],
                    )

                    but3 = gr.Button(i18n("Train model"), variant="primary")
                    but4 = gr.Button(i18n("Train feature index"), variant="primary")
                    but5 = gr.Button(i18n("One-click training"), variant="primary")
            with gr.Row():
                info3 = gr.Textbox(label=i18n("Output information"), value="")
                but3.click(
                    click_train,
                    [
                        exp_dir1,
                        sr2,
                        if_f0_3,
                        spk_id5,
                        save_epoch10,
                        total_epoch11,
                        batch_size12,
                        if_save_latest13,
                        pretrained_G14,
                        pretrained_D15,
                        gpus16,
                        if_cache_gpu17,
                        if_save_every_weights18,
                        version19,
                        author,
                    ],
                    info3,
                    api_name="train_start",
                )
                but4.click(train_index, [exp_dir1, version19], info3)
                but5.click(
                    train1key,
                    [
                        exp_dir1,
                        sr2,
                        if_f0_3,
                        trainset_dir4,
                        spk_id5,
                        np7,
                        f0method8,
                        save_epoch10,
                        total_epoch11,
                        batch_size12,
                        if_save_latest13,
                        pretrained_G14,
                        pretrained_D15,
                        gpus16,
                        if_cache_gpu17,
                        if_save_every_weights18,
                        version19,
                        author,
                    ],
                    info3,
                    api_name="train_start_all",
                )

        with gr.TabItem(i18n("ckpt Processing")):
            gr.Markdown(
                value=i18n(
                    "### Model comparison\n> You can get model ID (long) from `View model information` below.\n\nCalculate a similarity between two models."
                )
            )
            with gr.Row():
                with gr.Column():
                    id_a = gr.Textbox(label=i18n("ID of model A (long)"), value="")
                    id_b = gr.Textbox(label=i18n("ID of model B (long)"), value="")
                with gr.Column():
                    butmodelcmp = gr.Button(i18n("Calculate"), variant="primary")
                    infomodelcmp = gr.Textbox(
                        label=i18n("Similarity (from 0 to 1)"),
                        value="",
                        max_lines=1,
                    )
            butmodelcmp.click(
                hash_similarity,
                [
                    id_a,
                    id_b,
                ],
                infomodelcmp,
                api_name="ckpt_merge",
            )

            gr.Markdown(
                value=i18n("### Model fusion\nCan be used to test timbre fusion.")
            )
            with gr.Row():
                with gr.Column():
                    ckpt_a = gr.Textbox(
                        label=i18n("Path to Model A"), value="", interactive=True
                    )
                    ckpt_b = gr.Textbox(
                        label=i18n("Path to Model B"), value="", interactive=True
                    )
                    alpha_a = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("Weight (w) for Model A"),
                        value=0.5,
                        interactive=True,
                    )
                with gr.Column():
                    sr_ = gr.Radio(
                        label=i18n("Target sample rate"),
                        choices=["32k", "40k", "48k"],
                        value="48k",
                        interactive=True,
                    )
                    if_f0_ = gr.Radio(
                        label=i18n("Whether the model has pitch guidance"),
                        choices=[i18n("Yes"), i18n("No")],
                        value=i18n("Yes"),
                        interactive=True,
                    )
                    info__ = gr.Textbox(
                        label=i18n("Model information to be placed"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                with gr.Column():
                    name_to_save0 = gr.Textbox(
                        label=i18n("Saved model name (without extension)"),
                        value="",
                        max_lines=1,
                        interactive=True,
                    )
                    version_2 = gr.Radio(
                        label=i18n("Model architecture version"),
                        choices=["v1", "v2"],
                        value="v1",
                        interactive=True,
                    )
                    but6 = gr.Button(i18n("Fusion"), variant="primary")
            with gr.Row():
                info4 = gr.Textbox(label=i18n("Output information"), value="")
            but6.click(
                merge,
                [
                    ckpt_a,
                    ckpt_b,
                    alpha_a,
                    sr_,
                    if_f0_,
                    info__,
                    name_to_save0,
                    version_2,
                ],
                info4,
                api_name="ckpt_merge",
            )  # def merge(path1,path2,alpha1,sr,f0,info):

            gr.Markdown(
                value=i18n(
                    "### Modify model information\n> Only supported for small model files extracted from the 'weights' folder."
                )
            )
            with gr.Row():
                with gr.Column():
                    ckpt_path0 = gr.Textbox(
                        label=i18n("Path to Model"), value="", interactive=True
                    )
                    info_ = gr.Textbox(
                        label=i18n("Model information to be modified"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                    name_to_save1 = gr.Textbox(
                        label=i18n("Save file name (default: same as the source file)"),
                        value="",
                        max_lines=1,
                        interactive=True,
                    )
                with gr.Column():
                    but7 = gr.Button(i18n("Modify"), variant="primary")
                    info5 = gr.Textbox(label=i18n("Output information"), value="")
            but7.click(
                change_info,
                [ckpt_path0, info_, name_to_save1],
                info5,
                api_name="ckpt_modify",
            )

            gr.Markdown(
                value=i18n(
                    "### View model information\n> Only supported for small model files extracted from the 'weights' folder."
                )
            )
            with gr.Row():
                with gr.Column():
                    ckpt_path1 = gr.File(label=i18n("Path to Model"))
                    but8 = gr.Button(i18n("View"), variant="primary")
                with gr.Column():
                    info6 = gr.Textbox(label=i18n("Output information"), value="")
            but8.click(show_info, [ckpt_path1], info6, api_name="ckpt_show")

            gr.Markdown(
                value=i18n(
                    "### Model extraction\n> Enter the path of the large file model under the 'logs' folder.\n\nThis is useful if you want to stop training halfway and manually extract and save a small model file, or if you want to test an intermediate model."
                )
            )
            with gr.Row():
                with gr.Column():
                    ckpt_path2 = gr.Textbox(
                        label=i18n("Path to Model"),
                        value="E:\\codes\\py39\\logs\\mi-test_f0_48k\\G_23333.pth",
                        interactive=True,
                    )
                    save_name = gr.Textbox(
                        label=i18n("Save name"), value="", interactive=True
                    )
                    with gr.Row():
                        sr__ = gr.Radio(
                            label=i18n("Target sample rate"),
                            choices=["32k", "40k", "48k"],
                            value="48k",
                            interactive=True,
                        )
                        if_f0__ = gr.Radio(
                            label=i18n(
                                "Whether the model has pitch guidance (1: yes, 0: no)"
                            ),
                            choices=["1", "0"],
                            value="1",
                            interactive=True,
                        )
                        version_1 = gr.Radio(
                            label=i18n("Model architecture version"),
                            choices=["v1", "v2"],
                            value="v2",
                            interactive=True,
                        )
                    info___ = gr.Textbox(
                        label=i18n("Model information to be placed"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                    extauthor = gr.Textbox(
                        label=i18n("Model Author"),
                        value="",
                        max_lines=1,
                        interactive=True,
                    )
                with gr.Column():
                    but9 = gr.Button(i18n("Extract"), variant="primary")
                    info7 = gr.Textbox(label=i18n("Output information"), value="")
                    ckpt_path2.change(
                        change_info_, [ckpt_path2], [sr__, if_f0__, version_1]
                    )
            but9.click(
                extract_small_model,
                [
                    ckpt_path2,
                    save_name,
                    extauthor,
                    sr__,
                    if_f0__,
                    info___,
                    version_1,
                ],
                info7,
                api_name="ckpt_extract",
            )

        with gr.TabItem(i18n("Export Onnx")):
            with gr.Row():
                ckpt_dir = gr.Textbox(
                    label=i18n("RVC Model Path"), value="", interactive=True
                )
            with gr.Row():
                onnx_dir = gr.Textbox(
                    label=i18n("Onnx Export Path"), value="", interactive=True
                )
            with gr.Row():
                infoOnnx = gr.Label(label="info")
            with gr.Row():
                butOnnx = gr.Button(i18n("Export Onnx Model"), variant="primary")
            butOnnx.click(
                export_onnx, [ckpt_dir, onnx_dir], infoOnnx, api_name="export_onnx"
            )

        tab_faq = i18n("FAQ (Frequently Asked Questions)")
        with gr.TabItem(tab_faq):
            try:
                # Always load English FAQ
                with open("docs/en/faq_en.md", "r", encoding="utf8") as f:
                    info = f.read()
                gr.Markdown(value=info)
            except:
                gr.Markdown(traceback.format_exc())

def main():
    try:
        import signal

        def cleanup(signum, frame):
            signame = signal.Signals(signum).name
            print(f"Got signal {signame} ({signum})")
            app.close()
            sys.exit(0)

        signal.signal(signal.SIGINT, cleanup)
        signal.signal(signal.SIGTERM, cleanup)
        if config.listen_host is None:
            os.environ.pop("GRADIO_SERVER_NAME", None)
        if config.listen_port is None:
            os.environ.pop("GRADIO_SERVER_PORT", None)
        launch_kwargs = {
            "max_threads": 511,
            "inbrowser": not config.noautoopen,
            "quiet": True,
        }
        if config.global_link:
            launch_kwargs["share"] = True
        if config.listen_host:
            launch_kwargs["server_name"] = config.listen_host
        if config.listen_port is not None:
            launch_kwargs["server_port"] = config.listen_port
        queued_app = app.queue(max_size=1022)
        _, local_url, share_url = queued_app.launch(
            prevent_thread_lock=True, **launch_kwargs
        )
        _print_gradio_startup_urls(local_url, share_url)
        queued_app.block_thread()
    except Exception as e:
        logger.error(str(e))


if __name__ == "__main__":
    main()
