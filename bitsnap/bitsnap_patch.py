from __future__ import annotations

"""BitSnap monkey patches for Megatron-LM checkpointing.

Usage (early in your entrypoint, before training starts):

```python
import bitsnap.bitsnap_patch as bitsnap_patch
bitsnap_patch.apply()
```

This will replace:
- megatron.checkpointing.save_checkpoint
- megatron.checkpointing.load_checkpoint

So that Adam optimizer state tensors (`exp_avg`, `exp_avg_sq`) are stored as
BitSnap compressed objects inside `model_optim_rng.pt`, while the directory
layout and filenames remain unchanged.
"""

import os
import random
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch

from bitsnap.bitsnap_quantizer import BitSnapOptimizerQuantizer


_GLOBAL_QUANTIZER: Optional[BitSnapOptimizerQuantizer] = None


def _get_quantizer() -> BitSnapOptimizerQuantizer:
    global _GLOBAL_QUANTIZER
    if _GLOBAL_QUANTIZER is None:
        _GLOBAL_QUANTIZER = BitSnapOptimizerQuantizer(alpha=0.9)
    return _GLOBAL_QUANTIZER


def _rank() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except Exception:
        pass
    return 0


def _target_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            return torch.device("cuda", torch.cuda.current_device())
        except Exception:
            return torch.device("cuda")
    return torch.device("cpu")


def _is_bitsnap_obj(x: Any) -> bool:
    return isinstance(x, dict) and x.get("__bitsnap__", False) is True


def _compress_optimizer_states(optimizer_state_dict: Dict[str, Any]) -> None:
    """In-place: replace exp_avg/exp_avg_sq tensors with BitSnap objects."""
    quantizer = _get_quantizer()

    # Megatron wrapper structure (see megatron/optimizer/optimizer.py):
    # optimizer_state_dict['optimizer'] is torch.optim state_dict()
    base = optimizer_state_dict.get("optimizer")
    if not isinstance(base, dict):
        return

    state = base.get("state")
    if not isinstance(state, dict):
        return

    dry_run = os.getenv("BITSNAP_DRY_RUN", "0") == "1"
    max_logs = int(os.getenv("BITSNAP_DRY_RUN_MAX_TENSORS", "3"))
    logged = 0

    for _param_id, per_param in state.items():
        if not isinstance(per_param, dict):
            continue
        for k in ("exp_avg", "exp_avg_sq"):
            v = per_param.get(k)
            if not isinstance(v, torch.Tensor):
                continue
            # Only compress FP32 states (as requested).
            if v.dtype != torch.float32:
                continue

            if dry_run and logged < max_logs:
                try:
                    quantizer.dry_run(v)
                except Exception:
                    pass
                logged += 1

            per_param[k] = quantizer.compress(v)


def _decompress_optimizer_states(optimizer_state_dict: Dict[str, Any]) -> None:
    """In-place: replace BitSnap objects back to FP32 tensors."""
    quantizer = _get_quantizer()

    base = optimizer_state_dict.get("optimizer")
    if not isinstance(base, dict):
        return

    state = base.get("state")
    if not isinstance(state, dict):
        return

    device = _target_device()

    for _param_id, per_param in state.items():
        if not isinstance(per_param, dict):
            continue
        for k in ("exp_avg", "exp_avg_sq"):
            v = per_param.get(k)
            if _is_bitsnap_obj(v):
                per_param[k] = quantizer.decompress(v, device=device)


def save_checkpoint_bitsnap(iteration, model, optimizer, lr_scheduler):
    """BitSnap version of megatron.checkpointing.save_checkpoint (logic mirrored)."""
    # Import inside to avoid importing Megatron at module import time.
    from megatron import get_args, mpu, print_rank_0, update_num_microbatches, utils  # noqa: F401
    from megatron.checkpointing import (
        ensure_directory_exists,
        get_checkpoint_name,
        get_checkpoint_tracker_filename,
    )

    args = get_args()

    # Only rank zero of the data parallel writes to the disk.
    model = utils.unwrap_model(model)

    print_rank_0(f"saving checkpoint at iteration {iteration:7d} to {args.save}")

    if not torch.distributed.is_initialized() or mpu.get_data_parallel_rank() == 0:
        # Arguments, iteration, and model.
        state_dict: Dict[str, Any] = {}
        state_dict["args"] = args
        state_dict["checkpoint_version"] = 3.0
        state_dict["iteration"] = iteration

        if len(model) == 1:
            state_dict["model"] = model[0].state_dict_for_save_checkpoint()
        else:
            for i in range(len(model)):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                state_dict[f"model{i}"] = model[i].state_dict_for_save_checkpoint()

        # Optimizer stuff.
        if not args.no_save_optim:
            if optimizer is not None:
                state_dict["optimizer"] = optimizer.state_dict()
                # --- BitSnap hook: compress exp_avg/exp_avg_sq before torch.save ---
                try:
                    _compress_optimizer_states(state_dict["optimizer"])
                except Exception as e:
                    print_rank_0(f"[BitSnap] WARNING: optimizer compression failed: {e}")
            if lr_scheduler is not None:
                state_dict["lr_scheduler"] = lr_scheduler.state_dict()

        # RNG states.
        if not args.no_save_rng:
            state_dict["random_rng_state"] = random.getstate()
            state_dict["np_rng_state"] = np.random.get_state()
            state_dict["torch_rng_state"] = torch.get_rng_state()
            state_dict["cuda_rng_state"] = torch.cuda.get_rng_state()
            state_dict["rng_tracker_states"] = mpu.get_cuda_rng_tracker().get_states()

        # Save.
        checkpoint_name = get_checkpoint_name(args.save, iteration)
        ensure_directory_exists(checkpoint_name)
        torch.save(state_dict, checkpoint_name)

    # Wait so everyone is done (necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(f"  successfully saved checkpoint at iteration {iteration:7d} to {args.save}")

    # And update the latest iteration
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, "w") as f:
            f.write(str(iteration))

    # Wait so everyone is done (not necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def load_checkpoint_bitsnap(model, optimizer, lr_scheduler, load_arg="load", strict=True):
    """BitSnap version of megatron.checkpointing.load_checkpoint (logic mirrored)."""
    from megatron import get_args, mpu, print_rank_0, update_num_microbatches, utils
    from megatron.checkpointing import (
        check_checkpoint_args,
        fix_query_key_value_ordering,
        get_checkpoint_name,
        get_checkpoint_tracker_filename,
        read_metadata,
        set_checkpoint_version,
        get_checkpoint_version,
    )

    args = get_args()
    load_dir = getattr(args, load_arg)

    model = utils.unwrap_model(model)

    tracker_filename = get_checkpoint_tracker_filename(load_dir)
    if not os.path.isfile(tracker_filename):
        print_rank_0(f"WARNING: could not find the metadata file {tracker_filename} ")
        print_rank_0("    will not load any checkpoints and will start from random")
        return 0

    iteration, release = read_metadata(tracker_filename)
    checkpoint_name = get_checkpoint_name(load_dir, iteration, release)
    print_rank_0(f" loading checkpoint from {getattr(args, 'load', load_dir)} at iteration {iteration}")

    try:
        state_dict = torch.load(checkpoint_name, map_location="cpu")
    except ModuleNotFoundError:
        from megatron.fp16_deprecated import loss_scaler  # noqa: F401

        print_rank_0(" > deserializing using the old code structure ...")
        sys.modules["fp16.loss_scaler"] = sys.modules["megatron.fp16_deprecated.loss_scaler"]
        sys.modules["megatron.fp16.loss_scaler"] = sys.modules["megatron.fp16_deprecated.loss_scaler"]
        state_dict = torch.load(checkpoint_name, map_location="cpu")
        sys.modules.pop("fp16.loss_scaler", None)
        sys.modules.pop("megatron.fp16.loss_scaler", None)
    except BaseException as e:
        print_rank_0("could not load the checkpoint")
        print_rank_0(e)
        sys.exit()

    # set checkpoint version
    set_checkpoint_version(state_dict.get("checkpoint_version", 0))

    # Set iteration.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = state_dict["iteration"]
        except KeyError:
            try:
                iteration = state_dict["total_iters"]
            except KeyError:
                print_rank_0(
                    f"A metadata file exists but unable to load iteration from checkpoint {checkpoint_name}, exiting"
                )
                sys.exit()

    # Check arguments.
    assert args.consumed_train_samples == 0
    assert args.consumed_valid_samples == 0
    if "args" in state_dict:
        checkpoint_args = state_dict["args"]
        check_checkpoint_args(checkpoint_args)
        args.consumed_train_samples = getattr(checkpoint_args, "consumed_train_samples", 0)
        update_num_microbatches(consumed_samples=args.consumed_train_samples)
        args.consumed_valid_samples = getattr(checkpoint_args, "consumed_valid_samples", 0)
    else:
        print_rank_0("could not find arguments in the checkpoint ...")

    # Model.
    if len(model) == 1:
        model[0].load_state_dict(state_dict["model"], strict=strict)
    else:
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            model[i].load_state_dict(state_dict[f"model{i}"], strict=strict)

    # Fix up query/key/value matrix ordering if needed
    checkpoint_version = get_checkpoint_version()
    print_rank_0(f" checkpoint version {checkpoint_version}")
    fix_query_key_value_ordering(model, checkpoint_version)

    # Optimizer.
    if not release and not args.finetune and not args.no_load_optim:
        try:
            if optimizer is not None:
                # --- BitSnap hook: decompress exp_avg/exp_avg_sq before optimizer.load_state_dict ---
                try:
                    if "optimizer" in state_dict:
                        _decompress_optimizer_states(state_dict["optimizer"])
                except Exception as e:
                    print_rank_0(f"[BitSnap] WARNING: optimizer decompression failed: {e}")
                optimizer.load_state_dict(state_dict["optimizer"])
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        except KeyError:
            print_rank_0(
                f"Unable to load optimizer from checkpoint {checkpoint_name}. "
                "Specify --no-load-optim or --finetune to prevent attempting to load the optimizer state, exiting ..."
            )
            sys.exit()

    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            random.setstate(state_dict["random_rng_state"])
            np.random.set_state(state_dict["np_rng_state"])
            torch.set_rng_state(state_dict["torch_rng_state"])
            torch.cuda.set_rng_state(state_dict["cuda_rng_state"])
            if not state_dict["rng_tracker_states"]:
                raise KeyError
            mpu.get_cuda_rng_tracker().set_states(state_dict["rng_tracker_states"])
        except KeyError:
            print_rank_0(
                f"Unable to load rng state from checkpoint {checkpoint_name}. "
                "Specify --no-load-rng or --finetune to prevent attempting to load the rng state, exiting ..."
            )
            sys.exit()

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(f"  successfully loaded checkpoint from {args.load} at iteration {iteration}")
    return iteration


def apply() -> None:
    """Apply monkey patch: replace megatron.checkpointing save/load with BitSnap versions."""
    import megatron.checkpointing as ckpt

    ckpt.save_checkpoint = save_checkpoint_bitsnap
    ckpt.load_checkpoint = load_checkpoint_bitsnap

