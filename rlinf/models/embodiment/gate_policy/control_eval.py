# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Evaluation-only FastWAM mechanism controls.

These controls never extend the production binary WAMMode space.  They force a
single experimental branch for a closed-loop endpoint evaluation and take their
cost/solver settings from a separately profiled, provenance-bound artifact.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import yaml


CONTROL_KINDS = {
    "valid_idm",
    "no_read",
    "repeat_current",
    "shuffled",
    "extra_compute",
}
IDM_BRANCH_CONTROLS = CONTROL_KINDS - {"extra_compute"}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cfg_get(cfg: object | None, key: str, default=None):
    if cfg is None:
        return default
    getter = getattr(cfg, "get", None)
    return getter(key, default) if callable(getter) else default


def normalize_control_kind(value: object) -> str | None:
    if value is None or str(value).strip().lower() in {"", "none", "null", "off"}:
        return None
    result = str(value).strip().lower()
    if result not in CONTROL_KINDS:
        raise ValueError(
            f"unknown eval_control.kind={result!r}; expected one of "
            f"{sorted(CONTROL_KINDS)} or null"
        )
    return result


def configured_control_kind(cfg: object | None) -> str | None:
    return normalize_control_kind(_cfg_get(cfg, "kind", None))


def _require_number(mapping: Mapping[str, Any], key: str, *, source: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{source}.{key} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}.{key} must be numeric") from exc
    if not torch.isfinite(torch.tensor(result)) or result < 0:
        raise ValueError(f"{source}.{key} must be finite and non-negative")
    return result


def _require_seed(value: object, *, source: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{source} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be a non-negative integer") from exc
    try:
        exact = float(value) == float(result)
    except (TypeError, ValueError):
        exact = False
    if not exact or result < 0:
        raise ValueError(f"{source} must be a non-negative integer")
    return result


@dataclass(frozen=True)
class ControlProfile:
    path: str
    sha256: str
    metadata: dict[str, Any]
    controls: dict[str, dict[str, Any]]

    def entry(self, control: str) -> dict[str, Any]:
        try:
            return self.controls[control]
        except KeyError as exc:
            raise ValueError(
                f"control profile {self.path} has no {control!r} entry"
            ) from exc


def load_control_profile(
    path: str | Path,
    *,
    expected_metadata: Mapping[str, Any],
) -> ControlProfile:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"FastWAM control profile not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("control profile must be a mapping")
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported FastWAM control profile schema")
    if payload.get("kind") != "fastwam_control_profile":
        raise ValueError(
            f"unexpected control profile kind {payload.get('kind')!r}"
        )
    metadata = payload.get("meta")
    controls = payload.get("controls")
    if not isinstance(metadata, Mapping) or not isinstance(controls, Mapping):
        raise ValueError("control profile requires meta and controls mappings")
    required_metadata = {
        "task",
        "ckpt_fingerprint",
        "dataset_stats_fingerprint",
        "inference_steps",
        "solver_fingerprint",
        "height",
        "width",
        "num_video_frames",
        "action_horizon",
        "context_len",
        "model_dtype",
        "device_name",
    }
    missing_metadata = sorted(
        key for key in required_metadata if metadata.get(key) in (None, "")
    )
    if missing_metadata:
        raise ValueError(
            f"control profile metadata is missing required fields {missing_metadata}"
        )
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in expected_metadata.items()
        if expected is not None and metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "FastWAM control profile provenance mismatch (actual, expected): "
            f"{mismatches}"
        )
    normalized_controls: dict[str, dict[str, Any]] = {}
    for name, raw in controls.items():
        if str(name) not in CONTROL_KINDS or not isinstance(raw, Mapping):
            raise ValueError(f"malformed control profile entry {name!r}")
        entry = dict(raw)
        _require_number(entry, "flops", source=f"controls.{name}")
        _require_number(entry, "latency_ms", source=f"controls.{name}")
        raw_action_steps = entry.get("action_steps", 0)
        if isinstance(raw_action_steps, bool):
            raise ValueError(f"controls.{name}.action_steps must be positive")
        action_steps = int(raw_action_steps)
        if action_steps <= 0 or float(raw_action_steps) != float(action_steps):
            raise ValueError(
                f"controls.{name}.action_steps must be a positive integer"
            )
        if name != "extra_compute" and action_steps != int(
            metadata["inference_steps"]
        ):
            raise ValueError(
                f"controls.{name}.action_steps must equal profiled inference_steps"
            )
        entry["action_steps"] = action_steps
        normalized_controls[str(name)] = entry
    return ControlProfile(
        path=str(path),
        sha256=_sha256_file(path),
        metadata=dict(metadata),
        controls=normalized_controls,
    )


def _batched_value(value: object, index: int, default=None):
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1)[index].item()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value[index]
    return value


def recipient_control_metadata(
    context: Mapping[str, Any] | None, index: int
) -> tuple[str, dict[str, Any], bool]:
    if not isinstance(context, Mapping):
        raise ValueError("evaluation control requires obs.gate_context")
    uid = str(
        _batched_value(
            context.get("episode_uid", context.get("episode_key")), index, ""
        )
    )
    slot = int(_batched_value(context.get("decision_index"), index, -1))
    if not uid or slot < 0:
        raise ValueError(
            "shuffled control requires episode_uid and non-negative decision_index"
        )
    task = ""
    for key in ("base_task", "task_description", "task"):
        candidate = _batched_value(context.get(key), index, "")
        if candidate not in (None, ""):
            task = str(candidate)
            break
    cell = {
        "task": task,
        "factor": str(_batched_value(context.get("factor"), index, "")),
        "level": str(_batched_value(context.get("level"), index, "")),
        "phase": str(_batched_value(context.get("phase"), index, "")),
    }
    if any(not value or value == "unknown" for value in cell.values()):
        raise ValueError(
            f"shuffled control needs explicit task/factor/level/phase cell, got {cell}"
        )
    reliable = bool(_batched_value(context.get("phase_reliable"), index, False))
    return f"{uid}:{slot:03d}", cell, reliable


def _capture_context_metadata(
    context: Mapping[str, Any] | None,
    index: int,
) -> tuple[str, dict[str, Any]]:
    state_id, cell, phase_reliable = recipient_control_metadata(context, index)
    if not phase_reliable:
        raise ValueError(
            "donor capture requires a reliable pre-treatment phase label"
        )
    assert context is not None
    metadata = {
        **cell,
        "phase_reliable": True,
        "episode_uid": str(
            _batched_value(
                context.get("episode_uid", context.get("episode_key")), index, ""
            )
        ),
        "decision_index": int(
            _batched_value(context.get("decision_index"), index, -1)
        ),
        "task_id": str(_batched_value(context.get("task_id"), index, "")),
        "trial_id": str(_batched_value(context.get("trial_id"), index, "")),
        "reset_state_id": int(
            _batched_value(context.get("reset_state_id"), index, -1)
        ),
        "env_seed": int(_batched_value(context.get("env_seed"), index, -1)),
        "perturbation_id": str(
            _batched_value(context.get("perturbation_id"), index, "")
        ),
        "episode_manifest_sha256": str(
            _batched_value(context.get("episode_manifest_sha256"), index, "")
        ),
    }
    return state_id, metadata


def _write_donor_artifact(
    output_dir: str | Path,
    *,
    state_id: str,
    video_latents: torch.Tensor,
    metadata: Mapping[str, Any],
    overwrite: bool,
) -> dict[str, str]:
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stem = hashlib.sha256(state_id.encode("utf-8")).hexdigest()[:24]
    target = directory / f"donor_{stem}.pt"
    temporary = directory / f".{target.name}.{os.getpid()}.tmp"
    payload = {
        "schema_version": 1,
        "kind": "fastwam_shuffled_future_donor",
        "state_id": state_id,
        "video_latents": video_latents.detach().to(device="cpu"),
        "metadata": dict(metadata),
    }
    torch.save(payload, temporary)
    try:
        if overwrite:
            os.replace(temporary, target)
        else:
            try:
                os.link(temporary, target)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"donor artifact already exists for state {state_id!r}: {target}; "
                    "use a fresh capture directory or set capture_overwrite=true"
                ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(target), "sha256": _sha256_file(target)}


@dataclass
class EvalControlRuntime:
    kind: str
    profile: ControlProfile
    cost_metric: str
    cost: float
    action_steps: int
    wam_seed: int
    donor_seed: int
    donor_bank: Any = None
    donor_bank_path: str | None = None
    donor_bank_sha256: str | None = None
    expected_donor_wam_seed: int | None = None
    capture_donor_dir: str | None = None
    capture_overwrite: bool = False

    @property
    def branch_mode(self) -> int:
        return 0 if self.kind == "extra_compute" else 1

    @property
    def max_decisions_method(self) -> str:
        return f"control:{self.kind}"

    def provenance(self) -> dict[str, Any]:
        return {
            "method": self.max_decisions_method,
            "control": self.kind,
            "control_profile_path": self.profile.path,
            "control_profile_sha256": self.profile.sha256,
            "control_cost_metric": self.cost_metric,
            "control_cost": self.cost,
            "control_action_steps": self.action_steps,
            "solver_fingerprint": self.profile.metadata["solver_fingerprint"],
            "wam_seed": self.wam_seed,
            "donor_seed": self.donor_seed,
            "donor_bank_path": self.donor_bank_path,
            "donor_bank_sha256": self.donor_bank_sha256,
            "expected_donor_wam_seed": self.expected_donor_wam_seed,
            "capture_donor_dir": self.capture_donor_dir,
            "capture_overwrite": self.capture_overwrite,
            "donor_bank_metadata": (
                None
                if self.donor_bank is None
                else dict(getattr(self.donor_bank, "metadata", {}))
            ),
        }

    def validate_input_shape(self, input_image: torch.Tensor) -> None:
        expected_h = int(self.profile.metadata["height"])
        expected_w = int(self.profile.metadata["width"])
        actual = tuple(int(value) for value in input_image.shape[-2:])
        if actual != (expected_h, expected_w):
            raise ValueError(
                "control profile image shape mismatch: "
                f"expected {(expected_h, expected_w)}, got {actual}"
            )

    def act(
        self,
        adapter,
        *,
        input_image: torch.Tensor,
        proprio: torch.Tensor | None,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        encoded_state,
        gate_context: Mapping[str, Any] | None,
        batch_index: int,
    ) -> dict[str, Any]:
        self.validate_input_shape(input_image)
        active = bool(
            _batched_value(
                None if gate_context is None else gate_context.get("_active_mask"),
                batch_index,
                True,
            )
        )
        if not active:
            # Evaluation still queries the policy through the fixed horizon to
            # materialize learned reference schedules. Absorbed slots must not
            # create donor artifacts or invoke content interventions.
            result = dict(
                adapter.act(
                    input_image=input_image,
                    mode=self.branch_mode,
                    proprio=proprio,
                    context=context,
                    context_mask=context_mask,
                    encoded_state=encoded_state,
                    seed=self.wam_seed,
                )
            )
            result["cost"] = self.cost
            result["aux"] = {
                **dict(result.get("aux", {})),
                "control": self.kind,
                "control_skipped_after_absorption": True,
                "donor_artifact": None,
            }
            return result
        donor = None
        expected_donor_metadata = None
        if self.kind == "shuffled":
            if self.donor_bank is None:
                raise RuntimeError("shuffled control has no loaded donor bank")
            state_id, cell, phase_reliable = recipient_control_metadata(
                gate_context, batch_index
            )
            if not phase_reliable:
                raise ValueError(
                    "shuffled control requires a reliable pre-treatment phase label"
                )
            donor = self.donor_bank.select(
                recipient_state_id=state_id,
                recipient_metadata=cell,
                seed=self.donor_seed,
            )
            expected_donor_metadata = {
                **dict(getattr(self.donor_bank, "metadata", {})),
                **cell,
            }
        capture_state_id = None
        capture_metadata = None
        if self.capture_donor_dir is not None:
            capture_state_id, capture_metadata = _capture_context_metadata(
                gate_context, batch_index
            )
        result = adapter.act_control(
            input_image=input_image,
            control=self.kind,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            encoded_state=encoded_state,
            seed=self.wam_seed,
            shuffled_future_donor=donor,
            expected_donor_metadata=expected_donor_metadata,
            extra_action_steps=(
                self.action_steps if self.kind == "extra_compute" else None
            ),
            return_video_latents=(self.capture_donor_dir is not None),
        )
        result = dict(result)
        artifact = None
        if self.capture_donor_dir is not None:
            video_latents = result.pop("video_latents", None)
            if not torch.is_tensor(video_latents):
                raise RuntimeError(
                    "valid_idm donor capture requested, but FastWAM did not return "
                    "video_latents"
                )
            artifact = _write_donor_artifact(
                self.capture_donor_dir,
                state_id=str(capture_state_id),
                video_latents=video_latents,
                metadata={
                    **dict(capture_metadata or {}),
                    "ckpt_fingerprint": self.profile.metadata["ckpt_fingerprint"],
                    "dataset_stats_fingerprint": self.profile.metadata[
                        "dataset_stats_fingerprint"
                    ],
                    "solver_steps": int(self.profile.metadata["inference_steps"]),
                    "solver_fingerprint": self.profile.metadata[
                        "solver_fingerprint"
                    ],
                    "num_video_frames": int(
                        self.profile.metadata["num_video_frames"]
                    ),
                    "action_horizon": int(
                        self.profile.metadata["action_horizon"]
                    ),
                    "wam_seed": self.wam_seed,
                    "wam_task": self.profile.metadata["task"],
                    "control_profile_sha256": self.profile.sha256,
                },
                overwrite=self.capture_overwrite,
            )
        result["cost"] = self.cost
        result.setdefault("aux", {})
        result["aux"] = {
            **dict(result["aux"]),
            "control_profile_sha256": self.profile.sha256,
            "control_cost_metric": self.cost_metric,
            "donor_artifact": artifact,
        }
        return result


def _resolve_path(value: object, *, root: Path) -> Path:
    if value is None or str(value).lower() in {"", "none", "null"}:
        raise ValueError("required evaluation-control artifact path is unset")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def build_eval_control_runtime(
    cfg: object | None,
    *,
    adapter,
    fastwam_root: str | Path,
    donor_bank_loader: Callable[[Mapping[str, Any]], Any] | None = None,
) -> EvalControlRuntime | None:
    kind = configured_control_kind(cfg)
    if kind is None:
        return None
    if adapter is None:
        return None
    root = Path(fastwam_root).expanduser().resolve()
    profile_path = _resolve_path(_cfg_get(cfg, "profile_path", None), root=root)
    model = adapter.model
    expected_profile_metadata = {
        "task": str(adapter.task),
        "ckpt_fingerprint": getattr(model, "_loaded_checkpoint_fingerprint", None),
        "dataset_stats_fingerprint": getattr(
            adapter, "dataset_stats_fingerprint", None
        ),
        "inference_steps": int(adapter.inference_steps),
        "solver_fingerprint": getattr(adapter, "solver_fingerprint", None),
        "num_video_frames": int(adapter.num_video_frames),
        "action_horizon": int(adapter.generation_horizon),
        "context_len": int(adapter.context_len),
        "model_dtype": str(getattr(model, "torch_dtype", "")),
        "height": (getattr(adapter, "_cost_meta", None) or {}).get("height"),
        "width": (getattr(adapter, "_cost_meta", None) or {}).get("width"),
    }
    profile = load_control_profile(
        profile_path, expected_metadata=expected_profile_metadata
    )
    entry = profile.entry(kind)
    if kind in {"no_read", "extra_compute"} and bool(
        _cfg_get(cfg, "require_compute_matched", True)
    ) and not bool(entry.get("compute_matched", False)):
        raise ValueError(
            f"control profile marks {kind} as not compute-matched to valid IDM"
        )
    cost_metric = str(_cfg_get(cfg, "cost_metric", "latency_ms"))
    if cost_metric not in {"latency_ms", "flops"}:
        raise ValueError("eval_control.cost_metric must be latency_ms or flops")
    cost = _require_number(entry, cost_metric, source=f"controls.{kind}")
    action_steps = int(entry["action_steps"])
    if cost_metric == "latency_ms" and hasattr(adapter, "_device_name"):
        expected_device_name = adapter._device_name()
        actual_device_name = profile.metadata.get("device_name")
        if actual_device_name != expected_device_name:
            raise ValueError(
                "latency control profile hardware mismatch: "
                f"profile={actual_device_name!r}, current={expected_device_name!r}"
            )

    wam_seed = _require_seed(
        _cfg_get(cfg, "wam_seed", 0), source="eval_control.wam_seed"
    )
    donor_seed = _require_seed(
        _cfg_get(cfg, "donor_seed", 0), source="eval_control.donor_seed"
    )
    expected_donor_wam_seed = _require_seed(
        _cfg_get(cfg, "expected_donor_wam_seed", wam_seed),
        source="eval_control.expected_donor_wam_seed",
    )

    capture_dir = None
    capture_value = _cfg_get(cfg, "capture_donor_dir", None)
    if capture_value is not None and str(capture_value).lower() not in {
        "",
        "none",
        "null",
    }:
        if kind != "valid_idm":
            raise ValueError(
                "eval_control.capture_donor_dir is accepted only for valid_idm"
            )
        capture_path = _resolve_path(capture_value, root=root)
        capture_path.mkdir(parents=True, exist_ok=True)
        capture_dir = str(capture_path)
    capture_overwrite = bool(_cfg_get(cfg, "capture_overwrite", False))
    if capture_overwrite and capture_dir is None:
        raise ValueError(
            "eval_control.capture_overwrite requires capture_donor_dir"
        )

    donor_bank = None
    donor_path = None
    donor_sha = None
    raw_donor_path = _cfg_get(cfg, "donor_bank_path", None)
    if kind == "shuffled":
        donor_path_obj = _resolve_path(raw_donor_path, root=root)
        if not donor_path_obj.is_file():
            raise FileNotFoundError(
                f"shuffled-future donor bank not found: {donor_path_obj}"
            )
        payload = torch.load(
            donor_path_obj, map_location="cpu", weights_only=False
        )
        if not isinstance(payload, Mapping):
            raise ValueError("shuffled donor bank payload must be a mapping")
        if donor_bank_loader is None:
            from fastwam.adaptive_gate import ShuffledFutureBank

            donor_bank_loader = ShuffledFutureBank.from_payload
        donor_bank = donor_bank_loader(payload)
        expected_bank_metadata = {
            "ckpt_fingerprint": expected_profile_metadata["ckpt_fingerprint"],
            "dataset_stats_fingerprint": expected_profile_metadata[
                "dataset_stats_fingerprint"
            ],
            "solver_steps": expected_profile_metadata["inference_steps"],
            "solver_fingerprint": expected_profile_metadata[
                "solver_fingerprint"
            ],
            "num_video_frames": expected_profile_metadata["num_video_frames"],
            "action_horizon": expected_profile_metadata["action_horizon"],
            "wam_seed": expected_donor_wam_seed,
        }
        actual_bank_metadata = dict(getattr(donor_bank, "metadata", {}))
        mismatches = {
            key: (actual_bank_metadata.get(key), expected)
            for key, expected in expected_bank_metadata.items()
            if actual_bank_metadata.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                "shuffled donor bank WAM/stats/solver provenance mismatch "
                f"(actual, expected): {mismatches}"
            )
        donor_path = str(donor_path_obj)
        donor_sha = _sha256_file(donor_path_obj)
    elif raw_donor_path is not None and str(raw_donor_path).lower() not in {
        "",
        "none",
        "null",
    }:
        raise ValueError(f"{kind} does not accept eval_control.donor_bank_path")

    return EvalControlRuntime(
        kind=kind,
        profile=profile,
        cost_metric=cost_metric,
        cost=cost,
        action_steps=action_steps,
        wam_seed=wam_seed,
        donor_seed=donor_seed,
        donor_bank=donor_bank,
        donor_bank_path=donor_path,
        donor_bank_sha256=donor_sha,
        expected_donor_wam_seed=(
            expected_donor_wam_seed if kind == "shuffled" else None
        ),
        capture_donor_dir=capture_dir,
        capture_overwrite=capture_overwrite,
    )
