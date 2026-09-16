from __future__ import annotations

import json
import os
from dataclasses import dataclass
from tempfile import NamedTemporaryFile
from typing import Any

from p2p_protocol import (
    PROTOCOL_VERSION,
    Manifest,
    TensorUpdate,
    UpdateBatch,
    batch_from_dict,
    build_batch,
    compute_model_root,
    make_batch_id,
    sha256_bytes,
)


class ValidationError(Exception):
    pass


@dataclass
class ApplyResult:
    accepted: bool
    reason: str


class SyncCore:
    def __init__(
        self,
        *,
        manifest: Manifest,
        node_id: str,
        tensor_bytes: dict[str, bytes],
        max_payload_bytes: int,
    ) -> None:
        self.manifest = manifest
        self.node_id = node_id
        self.max_payload_bytes = max_payload_bytes
        self._specs = manifest.spec_map()
        self.tensor_bytes = dict(tensor_bytes)
        self.tensor_hashes = {name: sha256_bytes(data) for name, data in self.tensor_bytes.items()}
        self.root = compute_model_root(self.tensor_hashes)
        self.sequence = 0
        self.tip_batch_id = f"genesis:{self.root}"
        self.batches: dict[str, UpdateBatch] = {}

    def _validate_update(self, update: TensorUpdate) -> bytes:
        spec = self._specs.get(update.name)
        if spec is None:
            raise ValidationError(f"unknown tensor '{update.name}'")
        if tuple(update.shape) != spec.shape:
            raise ValidationError(f"shape mismatch for '{update.name}'")
        if update.dtype != spec.dtype:
            raise ValidationError(f"dtype mismatch for '{update.name}'")
        raw = update.decode_bytes()
        if len(raw) != update.byte_length:
            raise ValidationError(f"invalid payload length for '{update.name}'")
        if sha256_bytes(raw) != update.data_sha256:
            raise ValidationError(f"hash mismatch for '{update.name}'")
        return raw

    def _batch_payload_size(self, batch: UpdateBatch) -> int:
        return sum(update.byte_length for update in batch.updates)

    def create_local_batch(self, changed: dict[str, bytes]) -> UpdateBatch | None:
        updates: list[TensorUpdate] = []
        new_hashes = dict(self.tensor_hashes)

        for name in sorted(changed):
            data = changed[name]
            data_hash = sha256_bytes(data)
            if self.tensor_hashes.get(name) == data_hash:
                continue
            spec = self._specs.get(name)
            if spec is None:
                raise ValidationError(f"local change includes unknown tensor '{name}'")
            updates.append(TensorUpdate.from_bytes(name, spec.shape, spec.dtype, data))
            new_hashes[name] = data_hash

        if not updates:
            return None

        resulting_root = compute_model_root(new_hashes)
        batch = build_batch(
            architecture_id=self.manifest.architecture_id,
            origin_node=self.node_id,
            parent_batch_id=self.tip_batch_id,
            sequence=self.sequence + 1,
            resulting_root=resulting_root,
            updates=updates,
        )

        if self._batch_payload_size(batch) > self.max_payload_bytes:
            raise ValidationError("batch exceeds max payload size")

        self._apply_validated_batch(batch, decoded={u.name: u.decode_bytes() for u in batch.updates})
        return batch

    def validate_batch(self, batch: UpdateBatch) -> None:
        if batch.protocol_version != PROTOCOL_VERSION:
            raise ValidationError("protocol version mismatch")
        if batch.architecture_id != self.manifest.architecture_id:
            raise ValidationError("architecture mismatch")
        if batch.parent_batch_id != self.tip_batch_id:
            raise ValidationError("predecessor mismatch")
        if batch.sequence != self.sequence + 1:
            raise ValidationError("sequence mismatch")
        if self._batch_payload_size(batch) > self.max_payload_bytes:
            raise ValidationError("batch exceeds max payload size")

        payload = {
            "protocol_version": batch.protocol_version,
            "architecture_id": batch.architecture_id,
            "origin_node": batch.origin_node,
            "parent_batch_id": batch.parent_batch_id,
            "sequence": batch.sequence,
            "created_at": batch.created_at,
            "resulting_root": batch.resulting_root,
            "updates": [update.as_dict() for update in batch.updates],
        }
        expected_batch_id = make_batch_id(payload)
        if expected_batch_id != batch.batch_id:
            raise ValidationError("batch id hash mismatch")

        for update in batch.updates:
            self._validate_update(update)

    def accept_remote_batch(self, batch: UpdateBatch) -> ApplyResult:
        try:
            self.validate_batch(batch)
            decoded = {update.name: self._validate_update(update) for update in batch.updates}
            self._apply_validated_batch(batch, decoded)
        except ValidationError as exc:
            return ApplyResult(accepted=False, reason=str(exc))
        return ApplyResult(accepted=True, reason="accepted")

    def _apply_validated_batch(self, batch: UpdateBatch, decoded: dict[str, bytes]) -> None:
        new_hashes = dict(self.tensor_hashes)
        for name, data in decoded.items():
            self.tensor_bytes[name] = data
            new_hashes[name] = sha256_bytes(data)

        recomputed_root = compute_model_root(new_hashes)
        if recomputed_root != batch.resulting_root:
            raise ValidationError("resulting root mismatch")

        self.tensor_hashes = new_hashes
        self.root = recomputed_root
        self.tip_batch_id = batch.batch_id
        self.sequence = batch.sequence
        self.batches[batch.batch_id] = batch

    def tensor_chunk(self, name: str, offset: int, length: int) -> tuple[bytes, str]:
        data = self.tensor_bytes[name]
        chunk = data[offset : offset + length]
        return chunk, sha256_bytes(chunk)

    def apply_snapshot(
        self,
        *,
        tensor_bytes: dict[str, bytes],
        remote_tip_batch_id: str,
        remote_sequence: int,
        remote_root: str,
    ) -> None:
        new_hashes = {name: sha256_bytes(data) for name, data in tensor_bytes.items()}
        recomputed_root = compute_model_root(new_hashes)
        if recomputed_root != remote_root:
            raise ValidationError("snapshot root mismatch")
        self.tensor_bytes = dict(tensor_bytes)
        self.tensor_hashes = new_hashes
        self.root = recomputed_root
        self.tip_batch_id = remote_tip_batch_id
        self.sequence = remote_sequence

    def export_state(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.as_dict(),
            "tip_batch_id": self.tip_batch_id,
            "sequence": self.sequence,
            "root": self.root,
            "tensor_hashes": self.tensor_hashes,
            "batches": [batch.as_dict() for batch in self.batches.values()],
        }

    def atomic_persist(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=directory or None) as tmp:
            json.dump(self.export_state(), tmp, indent=2, sort_keys=True)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_name = tmp.name
        os.replace(tmp_name, path)

    @staticmethod
    def load_batch(data: dict[str, Any]) -> UpdateBatch:
        return batch_from_dict(data)
