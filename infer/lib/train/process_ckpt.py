import os
import traceback
from collections import OrderedDict
from time import time

import torch

from i18n.i18n import I18nAuto
from infer.modules.vc import model_hash_ckpt, hash_id

i18n = I18nAuto()


# add author sign
def save_small_model(ckpt, sr, if_f0, name, epoch, version, hps):
    try:
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].half()
        opt["config"] = [
            hps.data.filter_length // 2 + 1,
            32,
            hps.model.inter_channels,
            hps.model.hidden_channels,
            hps.model.filter_channels,
            hps.model.n_heads,
            hps.model.n_layers,
            hps.model.kernel_size,
            hps.model.p_dropout,
            hps.model.resblock,
            hps.model.resblock_kernel_sizes,
            hps.model.resblock_dilation_sizes,
            hps.model.upsample_rates,
            hps.model.upsample_initial_channel,
            hps.model.upsample_kernel_sizes,
            hps.model.spk_embed_dim,
            hps.model.gin_channels,
            hps.data.sampling_rate,
        ]
        opt["info"] = "%sepoch" % epoch
        opt["name"] = name
        opt["timestamp"] = int(time())
        if hps.author:
            opt["author"] = hps.author
        opt["sr"] = sr
        opt["f0"] = if_f0
        opt["version"] = version
        h = model_hash_ckpt(opt)
        opt["hash"] = h
        opt["id"] = hash_id(h)
        torch.save(opt, "assets/weights/%s.pth" % name)
        return "Success."
    except:
        return traceback.format_exc()


def extract_small_model(path, name, author, sr, if_f0, info, version):
    try:
        ckpt = torch.load(path, map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].half()
        if sr == "40k":
            opt["config"] = [
                1025,
                32,
                192,
                192,
                768,
                2,
                6,
                3,
                0,
                "1",
                [3, 7, 11],
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                [10, 10, 2, 2],
                512,
                [16, 16, 4, 4],
                109,
                256,
                40000,
            ]
        elif sr == "48k":
            if version == "v1":
                opt["config"] = [
                    1025,
                    32,
                    192,
                    192,
                    768,
                    2,
                    6,
                    3,
                    0,
                    "1",
                    [3, 7, 11],
                    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 6, 2, 2, 2],
                    512,
                    [16, 16, 4, 4, 4],
                    109,
                    256,
                    48000,
                ]
            else:
                opt["config"] = [
                    1025,
                    32,
                    192,
                    192,
                    768,
                    2,
                    6,
                    3,
                    0,
                    "1",
                    [3, 7, 11],
                    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [12, 10, 2, 2],
                    512,
                    [24, 20, 4, 4],
                    109,
                    256,
                    48000,
                ]
        elif sr == "32k":
            if version == "v1":
                opt["config"] = [
                    513,
                    32,
                    192,
                    192,
                    768,
                    2,
                    6,
                    3,
                    0,
                    "1",
                    [3, 7, 11],
                    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 4, 2, 2, 2],
                    512,
                    [16, 16, 4, 4, 4],
                    109,
                    256,
                    32000,
                ]
            else:
                opt["config"] = [
                    513,
                    32,
                    192,
                    192,
                    768,
                    2,
                    6,
                    3,
                    0,
                    "1",
                    [3, 7, 11],
                    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 8, 2, 2],
                    512,
                    [20, 16, 4, 4],
                    109,
                    256,
                    32000,
                ]
        if info == "":
            info = "Extracted model."
        opt["info"] = info
        opt["name"] = name
        opt["timestamp"] = int(time())
        if author:
            opt["author"] = author
        opt["version"] = version
        opt["sr"] = sr
        opt["f0"] = int(if_f0)
        h = model_hash_ckpt(opt)
        opt["hash"] = h
        opt["id"] = hash_id(h)
        torch.save(opt, "assets/weights/%s.pth" % name)
        return "Success."
    except:
        return traceback.format_exc()


def change_info(path, info, name):
    try:
        ckpt = torch.load(path, map_location="cpu")
        ckpt["info"] = info
        if name == "":
            name = os.path.basename(path)
        torch.save(ckpt, "assets/weights/%s" % name)
        return "Success."
    except:
        return traceback.format_exc()


def merge_many(paths, weights, sr, f0, info, name, version):
    try:
        def extract(ckpt):
            a = ckpt["model"]
            opt = OrderedDict()
            opt["weight"] = {}
            for key in a.keys():
                if "enc_q" in key:
                    continue
                opt["weight"][key] = a[key]
            return opt

        def authors(ckpts):
            all_authors = []
            for ckpt in ckpts:
                author = ckpt.get("author", "") or "Unknown"
                if author not in all_authors:
                    all_authors.append(author)
            if not all_authors:
                return ""
            return " & ".join(all_authors)

        if len(paths) < 2:
            return "Fail to merge the models. Please provide at least two models."
        if len(paths) != len(weights):
            return "Fail to merge the models. The number of weights does not match the number of models."

        normalized_weights = [float(weight) for weight in weights]
        if any(weight < 0 for weight in normalized_weights):
            return "Fail to merge the models. Weights must be non-negative."
        total_weight = sum(normalized_weights)
        if total_weight <= 0:
            return "Fail to merge the models. The total weight must be greater than zero."
        normalized_weights = [weight / total_weight for weight in normalized_weights]

        loaded_ckpts = [torch.load(path, map_location="cpu") for path in paths]
        cfg = loaded_ckpts[0]["config"]
        model_weights = []
        for ckpt in loaded_ckpts:
            if "model" in ckpt:
                model_weights.append(extract(ckpt)["weight"])
            else:
                model_weights.append(ckpt["weight"])
        reference_keys = sorted(list(model_weights[0].keys()))
        for current in model_weights[1:]:
            if sorted(list(current.keys())) != reference_keys:
                return "Fail to merge the models. The model architectures are not the same."
        opt = OrderedDict()
        opt["weight"] = {}
        for key in reference_keys:
            key_shapes = [tuple(current[key].shape) for current in model_weights]
            if key == "emb_g.weight" and len(set(key_shapes)) > 1:
                trailing_shapes = {shape[1:] for shape in key_shapes}
                if len(trailing_shapes) != 1:
                    return (
                        "Fail to merge the models. The speaker embedding shapes are not "
                        f"compatible for '{key}': {key_shapes}."
                    )
                min_shape0 = min(shape[0] for shape in key_shapes)
                merged_weight = None
                for index, current in enumerate(model_weights):
                    contribution = (
                        normalized_weights[index] * current[key][:min_shape0].float()
                    )
                    merged_weight = (
                        contribution
                        if merged_weight is None
                        else merged_weight + contribution
                    )
                opt["weight"][key] = merged_weight.half()
            else:
                if len(set(key_shapes)) != 1:
                    return (
                        "Fail to merge the models. Parameter "
                        f"'{key}' has different tensor shapes: {key_shapes}."
                    )
                merged_weight = None
                for index, current in enumerate(model_weights):
                    contribution = normalized_weights[index] * current[key].float()
                    merged_weight = (
                        contribution
                        if merged_weight is None
                        else merged_weight + contribution
                    )
                opt["weight"][key] = merged_weight.half()
        author = authors(loaded_ckpts)
        opt["config"] = cfg
        opt["name"] = name
        opt["timestamp"] = int(time())
        if author:
            opt["author"] = author
        opt["sr"] = sr
        if isinstance(f0, str):
            opt["f0"] = 1 if f0 == i18n("Yes") else 0
        else:
            opt["f0"] = 1 if f0 else 0
        opt["version"] = version
        opt["info"] = info
        h = model_hash_ckpt(opt)
        opt["hash"] = h
        opt["id"] = hash_id(h)
        torch.save(opt, "assets/weights/%s.pth" % name)
        return "Success."
    except:
        return traceback.format_exc()


def merge(path1, path2, alpha1, sr, f0, info, name, version):
    return merge_many(
        [path1, path2],
        [float(alpha1), 1 - float(alpha1)],
        sr,
        f0,
        info,
        name,
        version,
    )
