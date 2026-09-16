from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

PROTOCOL_VERSION = "1"


TRANSIENT_PREFIXES = ("embedtrace", "state.", "decaytrace.", "o.")


def is_transient_field(name: str) -> bool:
    return any(name == prefix or name.startswith(prefix) for prefix in TRANSIENT_PREFIXES)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class Manifest:
    protocol_version: str
    architecture_id: str
    tensor_specs: tuple[TensorSpec, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "architecture_id": self.architecture_id,
            "tensor_specs": [
                {"name": spec.name, "shape": list(spec.shape), "dtype": spec.dtype}
                for spec in self.tensor_specs
            ],
        }

    def spec_map(self) -> dict[str, TensorSpec]:
        return {spec.name: spec for spec in self.tensor_specs}

    @property
    def manifest_hash(self) -> str:
        canonical_specs = [
            {"name": spec.name, "shape": list(spec.shape), "dtype": spec.dtype}
            for spec in sorted(self.tensor_specs, key=lambda s: s.name)
        ]
        return sha256_bytes(_canonical_json(canonical_specs))


@dataclass(frozen=True)
class TensorUpdate:
    name: str
    shape: tuple[int, ...]
    dtype: str
    byte_length: int
    data_sha256: str
    data_b64: str

    @classmethod
    def from_bytes(cls, name: str, shape: tuple[int, ...], dtype: str, data: bytes) -> "TensorUpdate":
        return cls(
            name=name,
            shape=shape,
            dtype=dtype,
            byte_length=len(data),
            data_sha256=sha256_bytes(data),
            data_b64=base64.b64encode(data).decode("ascii"),
        )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["shape"] = list(self.shape)
        return result

    def decode_bytes(self) -> bytes:
        return base64.b64decode(self.data_b64.encode("ascii"))


@dataclass(frozen=True)
class UpdateBatch:
    protocol_version: str
    architecture_id: str
    origin_node: str
    parent_batch_id: str
    sequence: int
    created_at: str
    resulting_root: str
    updates: tuple[TensorUpdate, ...]
    batch_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "architecture_id": self.architecture_id,
            "origin_node": self.origin_node,
            "parent_batch_id": self.parent_batch_id,
            "sequence": self.sequence,
            "created_at": self.created_at,
            "resulting_root": self.resulting_root,
            "updates": [update.as_dict() for update in self.updates],
            "batch_id": self.batch_id,
        }


def compute_model_root(tensor_hashes: dict[str, str]) -> str:
    canonical = [{"name": name, "sha256": tensor_hashes[name]} for name in sorted(tensor_hashes)]
    return sha256_bytes(_canonical_json(canonical))


def make_batch_id(batch_without_id: dict[str, Any]) -> str:
    return sha256_bytes(_canonical_json(batch_without_id))


def build_batch(
    *,
    architecture_id: str,
    origin_node: str,
    parent_batch_id: str,
    sequence: int,
    resulting_root: str,
    updates: list[TensorUpdate],
) -> UpdateBatch:
    created_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "architecture_id": architecture_id,
        "origin_node": origin_node,
        "parent_batch_id": parent_batch_id,
        "sequence": sequence,
        "created_at": created_at,
        "resulting_root": resulting_root,
        "updates": [update.as_dict() for update in updates],
    }
    batch_id = make_batch_id(payload)
    return UpdateBatch(
        protocol_version=PROTOCOL_VERSION,
        architecture_id=architecture_id,
        origin_node=origin_node,
        parent_batch_id=parent_batch_id,
        sequence=sequence,
        created_at=created_at,
        resulting_root=resulting_root,
        updates=tuple(updates),
        batch_id=batch_id,
    )


def batch_from_dict(data: dict[str, Any]) -> UpdateBatch:
    updates = tuple(
        TensorUpdate(
            name=item["name"],
            shape=tuple(item["shape"]),
            dtype=item["dtype"],
            byte_length=item["byte_length"],
            data_sha256=item["data_sha256"],
            data_b64=item["data_b64"],
        )
        for item in data["updates"]
    )
    return UpdateBatch(
        protocol_version=data["protocol_version"],
        architecture_id=data["architecture_id"],
        origin_node=data["origin_node"],
        parent_batch_id=data["parent_batch_id"],
        sequence=int(data["sequence"]),
        created_at=data["created_at"],
        resulting_root=data["resulting_root"],
        updates=updates,
        batch_id=data["batch_id"],
    )
