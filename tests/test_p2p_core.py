from p2p_core import SyncCore
from p2p_protocol import Manifest, TensorSpec


def _manifest() -> Manifest:
    return Manifest(
        protocol_version="1",
        architecture_id="arch-1",
        tensor_specs=(
            TensorSpec(name="layers.0.weights.weight", shape=(2, 2), dtype="float32"),
            TensorSpec(name="decoder.decode.weight", shape=(2, 2), dtype="float32"),
        ),
    )


def _initial() -> dict[str, bytes]:
    return {
        "layers.0.weights.weight": (b"\x00\x00\x00\x00" * 4),
        "decoder.decode.weight": (b"\x01\x00\x00\x00" * 4),
    }


def test_create_and_accept_batch_round_trip() -> None:
    leader = SyncCore(manifest=_manifest(), node_id="n1", tensor_bytes=_initial(), max_payload_bytes=1024)
    follower = SyncCore(manifest=_manifest(), node_id="n2", tensor_bytes=_initial(), max_payload_bytes=1024)

    changed = {"layers.0.weights.weight": b"\x02\x00\x00\x00" * 4}
    batch = leader.create_local_batch(changed)

    assert batch is not None
    accepted = follower.accept_remote_batch(batch)
    assert accepted.accepted
    assert follower.root == leader.root
    assert follower.tensor_bytes["layers.0.weights.weight"] == changed["layers.0.weights.weight"]


def test_rejects_architecture_mismatch() -> None:
    leader = SyncCore(manifest=_manifest(), node_id="n1", tensor_bytes=_initial(), max_payload_bytes=1024)

    wrong_manifest = Manifest(
        protocol_version="1",
        architecture_id="other",
        tensor_specs=_manifest().tensor_specs,
    )
    follower = SyncCore(manifest=wrong_manifest, node_id="n2", tensor_bytes=_initial(), max_payload_bytes=1024)

    batch = leader.create_local_batch({"layers.0.weights.weight": b"\x03\x00\x00\x00" * 4})
    assert batch is not None

    result = follower.accept_remote_batch(batch)
    assert not result.accepted
    assert "architecture mismatch" in result.reason


def test_rejects_corrupt_payload_hash() -> None:
    leader = SyncCore(manifest=_manifest(), node_id="n1", tensor_bytes=_initial(), max_payload_bytes=1024)
    follower = SyncCore(manifest=_manifest(), node_id="n2", tensor_bytes=_initial(), max_payload_bytes=1024)

    batch = leader.create_local_batch({"layers.0.weights.weight": b"\x04\x00\x00\x00" * 4})
    assert batch is not None

    tampered = batch.as_dict()
    tampered["updates"][0]["data_b64"] = "AAAA"
    tampered_batch = SyncCore.load_batch(tampered)

    result = follower.accept_remote_batch(tampered_batch)
    assert not result.accepted
    assert "batch id hash mismatch" in result.reason or "hash mismatch" in result.reason
