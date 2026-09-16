from p2p_protocol import Manifest, TensorSpec, compute_model_root, is_transient_field


def test_model_root_stable_with_different_dict_order() -> None:
    hashes_a = {"b": "2" * 64, "a": "1" * 64}
    hashes_b = {"a": "1" * 64, "b": "2" * 64}
    assert compute_model_root(hashes_a) == compute_model_root(hashes_b)


def test_manifest_hash_stable() -> None:
    manifest_1 = Manifest(
        protocol_version="1",
        architecture_id="arch",
        tensor_specs=(
            TensorSpec(name="z", shape=(2, 2), dtype="float32"),
            TensorSpec(name="a", shape=(2,), dtype="float16"),
        ),
    )
    manifest_2 = Manifest(
        protocol_version="1",
        architecture_id="arch",
        tensor_specs=(
            TensorSpec(name="a", shape=(2,), dtype="float16"),
            TensorSpec(name="z", shape=(2, 2), dtype="float32"),
        ),
    )
    assert manifest_1.manifest_hash == manifest_2.manifest_hash


def test_transient_fields_are_excluded() -> None:
    assert is_transient_field("embedtrace")
    assert is_transient_field("state.0")
    assert is_transient_field("decaytrace.10")
    assert is_transient_field("o.encoder.embed.weight")
    assert not is_transient_field("layers.0.weights.weight")
