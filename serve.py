from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode, urlparse, urlunparse

import httpx
import numpy as np
import uvicorn
import websockets
from fastapi import FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from p2p_core import ApplyResult, SyncCore
from p2p_protocol import Manifest, PROTOCOL_VERSION, TensorSpec

LOGGER = logging.getLogger("p2p-sync")


@dataclass
class TrainingMetrics:
    enabled: bool = False
    running: bool = False
    epoch: int = 0
    step: int = 0
    examples_processed: int = 0
    bytes_processed: int = 0
    recent_loss: float | None = None
    losses: list[float] = field(default_factory=list)
    throughput_bytes_per_sec: float = 0.0
    last_eval_mcc: float | None = None
    last_eval_loss: float | None = None
    last_eval_confusion: dict[str, int] = field(default_factory=dict)


@dataclass
class PeerHealth:
    url: str
    connected: bool = False
    last_seen: float | None = None
    failures: int = 0
    last_error: str | None = None


class EventLog:
    def __init__(self, max_items: int = 500) -> None:
        self._events: deque[dict[str, Any]] = deque(maxlen=max_items)
        self._subscribers: set[asyncio.Queue[str]] = set()
        self._lock = threading.Lock()

    def add(self, level: str, kind: str, message: str, details: dict[str, Any] | None = None) -> None:
        event = {
            "time": time.time(),
            "level": level,
            "kind": kind,
            "message": message,
            "details": details or {},
        }
        with self._lock:
            self._events.append(event)
            payload = json.dumps(event)
            for queue in list(self._subscribers):
                queue.put_nowait(payload)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)

    async def stream(self):
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
        self._subscribers.add(queue)
        try:
            while True:
                payload = await queue.get()
                yield f"data: {payload}\n\n"
        finally:
            self._subscribers.discard(queue)


class ModelAdapter:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.lock = threading.Lock()

    def _flatten(self) -> dict[str, Any]:
        import mlx.utils as util

        return dict(util.tree_flatten(self.model.trainable_parameters()))

    def _tensor_bytes(self, tensor: Any) -> bytes:
        import mlx.core as mx

        mx.eval(tensor)
        return np.asarray(tensor).tobytes()

    def _tensor_from_bytes(self, data: bytes, shape: tuple[int, ...], dtype: str) -> Any:
        import mlx.core as mx

        array = np.frombuffer(data, dtype=np.dtype(dtype)).copy().reshape(shape)
        return mx.array(array)

    def manifest_and_state(self, architecture_id: str) -> tuple[Manifest, dict[str, bytes]]:
        with self.lock:
            flat = self._flatten()
            specs: list[TensorSpec] = []
            state: dict[str, bytes] = {}
            for name in sorted(flat):
                tensor = flat[name]
                shape = tuple(int(v) for v in tensor.shape)
                dtype = str(np.asarray(tensor).dtype)
                specs.append(TensorSpec(name=name, shape=shape, dtype=dtype))
                state[name] = self._tensor_bytes(tensor)
            manifest = Manifest(
                protocol_version=PROTOCOL_VERSION,
                architecture_id=architecture_id,
                tensor_specs=tuple(specs),
            )
            return manifest, state

    def current_state_bytes(self) -> dict[str, bytes]:
        with self.lock:
            flat = self._flatten()
            return {name: self._tensor_bytes(tensor) for name, tensor in flat.items()}

    def apply_tensor_bytes(self, tensor_bytes: dict[str, bytes], manifest: Manifest) -> None:
        import mlx.core as mx
        import mlx.utils as util

        with self.lock:
            flat = self._flatten()
            for spec in manifest.tensor_specs:
                data = tensor_bytes.get(spec.name)
                if data is None:
                    continue
                flat[spec.name] = self._tensor_from_bytes(data, spec.shape, spec.dtype)
            self.model.update(util.tree_unflatten(list(flat.items())))
            mx.eval(self.model.parameters())


class SyncNode:
    def __init__(
        self,
        *,
        model: Any,
        checkpoint_path: str,
        host: str,
        port: int,
        peers: list[str],
        sync_interval: float,
        max_payload_bytes: int,
        chunk_size: int,
        auth_token: str | None,
        train_text_path: str | None,
        train_loop: bool,
        cola_train_path: str | None,
        cola_eval_path: str | None,
        cola_epochs: int,
        run_cola: bool,
    ) -> None:
        self.node_id = str(uuid.uuid4())
        self.host = host
        self.port = port
        self.protocol_version = PROTOCOL_VERSION
        self.sync_interval = sync_interval
        self.max_payload_bytes = max_payload_bytes
        self.chunk_size = chunk_size
        self.auth_token = auth_token
        self.checkpoint_path = checkpoint_path
        self.train_text_path = train_text_path
        self.train_loop = train_loop
        self.cola_train_path = cola_train_path
        self.cola_eval_path = cola_eval_path
        self.cola_epochs = cola_epochs
        self.run_cola = run_cola
        self.model_adapter = ModelAdapter(model)
        self.events = EventLog()
        self.training = TrainingMetrics(enabled=train_loop or run_cola)
        self.last_sync_time: float | None = None
        self.last_error: str | None = None
        self.local_batch_count = 0
        self.remote_batch_count = 0
        self.peer_health: dict[str, PeerHealth] = {url: PeerHealth(url=url) for url in peers}
        self.peers = peers
        self.inbound_peers: set[str] = set()
        self.stop_event = asyncio.Event()
        self._tasks: list[asyncio.Task[Any]] = []
        self._peer_connections: dict[str, websockets.WebSocketClientProtocol] = {}
        self._batch_lock = asyncio.Lock()

        architecture_id = f"dim={model.dim}|layers={model.layercount}|manifest={self._manifest_hash()}"
        manifest, tensor_state = self.model_adapter.manifest_and_state(architecture_id)
        self.manifest = manifest
        self.core = SyncCore(
            manifest=manifest,
            node_id=self.node_id,
            tensor_bytes=tensor_state,
            max_payload_bytes=max_payload_bytes,
        )

    def _manifest_hash(self) -> str:
        manifest, _ = self.model_adapter.manifest_and_state("temp")
        return manifest.manifest_hash

    def _check_auth_header(self, value: str | None) -> None:
        if self.auth_token and value != self.auth_token:
            raise HTTPException(status_code=401, detail="invalid auth token")

    def _check_auth_ws(self, websocket: WebSocket) -> None:
        if self.auth_token and websocket.query_params.get("token") != self.auth_token:
            raise HTTPException(status_code=401, detail="invalid auth token")

    def status_payload(self) -> dict[str, Any]:
        now = time.time()
        return {
            "node": {
                "node_id": self.node_id,
                "protocol_version": self.protocol_version,
                "host": self.host,
                "port": self.port,
                "checkpoint_path": self.checkpoint_path,
                "architecture_id": self.manifest.architecture_id,
                "tensor_count": len(self.manifest.tensor_specs),
            },
            "training": asdict(self.training),
            "sync": {
                "model_root": self.core.root,
                "tip_batch_id": self.core.tip_batch_id,
                "sequence": self.core.sequence,
                "local_batches": self.local_batch_count,
                "remote_batches": self.remote_batch_count,
                "last_sync_time": self.last_sync_time,
                "last_error": self.last_error,
                "connected_outbound_peers": [url for url, health in self.peer_health.items() if health.connected],
                "connected_inbound_peers": sorted(self.inbound_peers),
                "peers": {
                    url: {
                        "connected": health.connected,
                        "last_seen": health.last_seen,
                        "seconds_since_seen": None if health.last_seen is None else now - health.last_seen,
                        "failures": health.failures,
                        "last_error": health.last_error,
                    }
                    for url, health in self.peer_health.items()
                },
                "share_policy": {
                    "shared": "trainable model parameter tensors only",
                    "not_shared": ["embedtrace", "state.*", "decaytrace.*", "optimizer state (o.*)"],
                },
            },
            "events": self.events.list()[-100:],
        }

    async def start(self) -> None:
        self.events.add("info", "startup", "Node started", {"node_id": self.node_id})
        self._tasks.append(asyncio.create_task(self._periodic_sync_task(), name="periodic-sync"))
        self._tasks.extend(asyncio.create_task(self._peer_loop(peer), name=f"peer-{peer}") for peer in self.peers)
        if self.train_loop:
            self._tasks.append(asyncio.create_task(self._training_task(), name="training"))
        if self.run_cola:
            self._tasks.append(asyncio.create_task(self._cola_task(), name="cola"))

    async def shutdown(self) -> None:
        self.stop_event.set()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for ws in self._peer_connections.values():
            with contextlib.suppress(Exception):
                await ws.close()

    async def _periodic_sync_task(self) -> None:
        while not self.stop_event.is_set():
            await asyncio.sleep(self.sync_interval)
            try:
                await self.flush_local_changes()
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                self.events.add("error", "sync", "Periodic sync failed", {"error": str(exc)})

    async def flush_local_changes(self) -> None:
        async with self._batch_lock:
            current = self.model_adapter.current_state_bytes()
            changed = {
                name: data
                for name, data in current.items()
                if self.core.tensor_hashes.get(name) != self._hash_bytes(data)
            }
            batch = self.core.create_local_batch(changed)
            if batch is None:
                return
            self.local_batch_count += 1
            self.last_sync_time = time.time()
            self.events.add(
                "info",
                "sync",
                "Created local update batch",
                {"batch_id": batch.batch_id, "updates": len(batch.updates), "sequence": batch.sequence},
            )
            await self._broadcast_json({"type": "batch", "batch": batch.as_dict()})

    def _hash_bytes(self, data: bytes) -> str:
        from p2p_protocol import sha256_bytes

        return sha256_bytes(data)

    async def _broadcast_json(self, payload: dict[str, Any]) -> None:
        msg = json.dumps(payload)
        for peer_url, ws in list(self._peer_connections.items()):
            try:
                await ws.send(msg)
                self.peer_health[peer_url].last_seen = time.time()
            except Exception as exc:  # noqa: BLE001
                self.peer_health[peer_url].last_error = str(exc)
                self.peer_health[peer_url].connected = False

    async def _peer_loop(self, peer_url: str) -> None:
        backoff = 1.0
        health = self.peer_health[peer_url]
        auth_query = {}
        if self.auth_token:
            auth_query["token"] = self.auth_token
        parsed = urlparse(peer_url)
        query = parsed.query
        extra = urlencode(auth_query)
        if extra:
            query = f"{query}&{extra}" if query else extra
        connect_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, query, parsed.fragment))

        while not self.stop_event.is_set():
            try:
                async with websockets.connect(connect_url, max_size=self.max_payload_bytes, open_timeout=10, ping_interval=15) as ws:
                    self._peer_connections[peer_url] = ws
                    health.connected = True
                    health.last_error = None
                    backoff = 1.0
                    await ws.send(
                        json.dumps(
                            {
                                "type": "hello",
                                "node_id": self.node_id,
                                "protocol_version": self.protocol_version,
                                "architecture_id": self.manifest.architecture_id,
                                "tip_batch_id": self.core.tip_batch_id,
                                "sequence": self.core.sequence,
                                "root": self.core.root,
                            }
                        )
                    )
                    self.events.add("info", "peer", "Connected to peer", {"peer": peer_url})
                    async for raw in ws:
                        if len(raw) > self.max_payload_bytes * 2:
                            raise ValueError("peer message exceeded max size")
                        await self._handle_peer_message(peer_url, raw)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                health.connected = False
                health.failures += 1
                health.last_error = str(exc)
                self.events.add("warning", "peer", "Peer connection failed", {"peer": peer_url, "error": str(exc)})
                await asyncio.sleep(min(backoff, 30.0))
                backoff = min(backoff * 2.0, 30.0)
            finally:
                self._peer_connections.pop(peer_url, None)
                health.connected = False

    async def _handle_peer_message(self, peer_url: str, raw: str) -> None:
        health = self.peer_health[peer_url]
        health.last_seen = time.time()
        msg = json.loads(raw)
        msg_type = msg.get("type")

        if msg_type == "hello":
            if msg.get("protocol_version") != self.protocol_version:
                self.events.add("warning", "validation", "Rejected hello with protocol mismatch", {"peer": peer_url})
                return
            if msg.get("architecture_id") != self.manifest.architecture_id:
                self.events.add("warning", "validation", "Rejected hello with architecture mismatch", {"peer": peer_url})
                return
            await self._broadcast_json(
                {
                    "type": "tip",
                    "tip_batch_id": self.core.tip_batch_id,
                    "sequence": self.core.sequence,
                    "root": self.core.root,
                }
            )
            return

        if msg_type == "tip":
            remote_sequence = int(msg.get("sequence", -1))
            if remote_sequence > self.core.sequence:
                await self._attempt_repair(peer_url)
            return

        if msg_type == "batch":
            batch = SyncCore.load_batch(msg["batch"])
            result: ApplyResult = self.core.accept_remote_batch(batch)
            if not result.accepted:
                self.events.add(
                    "warning",
                    "validation",
                    "Rejected remote batch",
                    {"peer": peer_url, "batch_id": batch.batch_id, "reason": result.reason},
                )
                if "predecessor mismatch" in result.reason or "sequence mismatch" in result.reason:
                    await self._attempt_repair(peer_url)
                return

            self.remote_batch_count += 1
            self.last_sync_time = time.time()
            self.model_adapter.apply_tensor_bytes(self.core.tensor_bytes, self.manifest)
            self.events.add(
                "info",
                "sync",
                "Applied remote update batch",
                {"peer": peer_url, "batch_id": batch.batch_id, "sequence": batch.sequence},
            )

    def _peer_http_base(self, peer_ws_url: str) -> str:
        parsed = urlparse(peer_ws_url)
        scheme = "https" if parsed.scheme == "wss" else "http"
        return urlunparse((scheme, parsed.netloc, "", "", "", "")).rstrip("/")

    async def _attempt_repair(self, peer_url: str) -> None:
        base = self._peer_http_base(peer_url)
        headers = {"x-sync-token": self.auth_token} if self.auth_token else {}

        async with httpx.AsyncClient(timeout=10.0) as client:
            manifest_resp = await client.get(f"{base}/sync/manifest", headers=headers)
            manifest_resp.raise_for_status()
            remote_manifest = manifest_resp.json()

            if remote_manifest["protocol_version"] != self.protocol_version:
                self.events.add("warning", "repair", "Repair aborted due to protocol mismatch", {"peer": peer_url})
                return
            if remote_manifest["architecture_id"] != self.manifest.architecture_id:
                self.events.add("warning", "repair", "Repair aborted due to architecture mismatch", {"peer": peer_url})
                return

            state_resp = await client.get(f"{base}/sync/state", headers=headers)
            state_resp.raise_for_status()
            remote_state = state_resp.json()
            remote_hashes = remote_state["tensor_hashes"]

            repaired = dict(self.core.tensor_bytes)
            for name, remote_hash in remote_hashes.items():
                if self.core.tensor_hashes.get(name) == remote_hash:
                    continue
                chunks: list[bytes] = []
                total_size = int(remote_state["tensor_sizes"][name])
                offset = 0
                while offset < total_size:
                    length = min(self.chunk_size, total_size - offset)
                    chunk_resp = await client.get(
                        f"{base}/sync/tensor/{name}",
                        params={"offset": offset, "length": length},
                        headers=headers,
                    )
                    chunk_resp.raise_for_status()
                    payload = chunk_resp.json()
                    data = base64.b64decode(payload["data_b64"])
                    local_hash = self._hash_bytes(data)
                    if local_hash != payload["chunk_sha256"]:
                        raise ValueError(f"chunk hash mismatch for tensor {name}")
                    chunks.append(data)
                    offset += len(data)
                repaired[name] = b"".join(chunks)

            self.core.apply_snapshot(
                tensor_bytes=repaired,
                remote_tip_batch_id=remote_state["tip_batch_id"],
                remote_sequence=int(remote_state["sequence"]),
                remote_root=remote_state["root"],
            )
            self.model_adapter.apply_tensor_bytes(self.core.tensor_bytes, self.manifest)
            self.last_sync_time = time.time()
            self.events.add(
                "info",
                "repair",
                "Completed snapshot repair from peer",
                {"peer": peer_url, "sequence": self.core.sequence, "root": self.core.root},
            )

    async def _training_task(self) -> None:
        path = self.train_text_path
        if path is None:
            self.events.add("warning", "training", "No training text path configured; training loop disabled")
            return
        text_path = Path(path)
        if not text_path.exists():
            self.events.add("error", "training", "Training path does not exist", {"path": path})
            return

        self.training.running = True
        started = time.time()
        self.events.add("info", "training", "Background training started", {"path": path})

        try:
            epoch = 0
            while not self.stop_event.is_set():
                epoch += 1
                self.training.epoch = epoch
                with text_path.open("r", encoding="utf-8", errors="ignore") as handle:
                    for line in handle:
                        if self.stop_event.is_set():
                            return
                        data = line.encode("utf-8")
                        if len(data) < 2:
                            continue
                        for idx, (currb, nextb) in enumerate(zip(data[:-1], data[1:])):
                            with self.model_adapter.lock:
                                self.model_adapter.model(currb, nextb, idx == len(data) - 2)
                            self.training.step += 1
                            self.training.examples_processed += 1
                            self.training.bytes_processed += 1

                            elapsed = max(time.time() - started, 1e-9)
                            self.training.throughput_bytes_per_sec = self.training.bytes_processed / elapsed

                            if self.training.step % 200 == 0:
                                self.events.add(
                                    "info",
                                    "training",
                                    "Training progress",
                                    {
                                        "epoch": self.training.epoch,
                                        "step": self.training.step,
                                        "bytes": self.training.bytes_processed,
                                        "throughput": self.training.throughput_bytes_per_sec,
                                    },
                                )
                        if self.training.step % 500 == 0:
                            with self.model_adapter.lock:
                                self.model_adapter.model.save(self.checkpoint_path)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            self.events.add("error", "training", "Training task failed", {"error": str(exc)})
        finally:
            self.training.running = False
            with self.model_adapter.lock:
                self.model_adapter.model.save(self.checkpoint_path)
            self.events.add("info", "training", "Background training stopped")

    async def _cola_task(self) -> None:
        from benchmark import evaluate_cola_head, load_cola, train_cola_head

        if not self.cola_train_path:
            self.events.add("error", "cola", "CoLA mode enabled but --cola-train-path not set")
            return

        self.training.running = True
        self.events.add("info", "cola", "CoLA workflow started")

        def progress_cb(update: dict[str, Any]) -> None:
            if update["phase"] == "train":
                self.training.epoch = int(update["epoch"])
                self.training.step = int(update["step"])
                self.training.recent_loss = float(update["loss"])
            else:
                self.training.last_eval_loss = float(update["loss"])
                self.training.last_eval_mcc = float(update["mcc"])
                self.training.last_eval_confusion = {
                    "tp": int(update["tp"]),
                    "tn": int(update["tn"]),
                    "fp": int(update["fp"]),
                    "fn": int(update["fn"]),
                }
            self.events.add("info", "cola", "CoLA progress", update)

        try:
            train_data = load_cola(self.cola_train_path)
            with self.model_adapter.lock:
                head, metrics = train_cola_head(
                    self.model_adapter.model,
                    train_data,
                    epochs=self.cola_epochs,
                    progress_cb=progress_cb,
                )

            self.training.recent_loss = metrics.loss
            self.training.last_eval_mcc = metrics.mcc
            self.training.last_eval_confusion = {
                "tp": metrics.confusion.tp,
                "tn": metrics.confusion.tn,
                "fp": metrics.confusion.fp,
                "fn": metrics.confusion.fn,
            }

            if self.cola_eval_path:
                eval_data = load_cola(self.cola_eval_path)
                with self.model_adapter.lock:
                    eval_metrics = evaluate_cola_head(
                        self.model_adapter.model,
                        head,
                        eval_data,
                        progress_cb=progress_cb,
                    )
                self.training.last_eval_loss = eval_metrics.loss
                self.training.last_eval_mcc = eval_metrics.mcc
                self.training.last_eval_confusion = {
                    "tp": eval_metrics.confusion.tp,
                    "tn": eval_metrics.confusion.tn,
                    "fp": eval_metrics.confusion.fp,
                    "fn": eval_metrics.confusion.fn,
                }

            self.events.add(
                "info",
                "cola",
                "CoLA workflow finished",
                {
                    "loss": self.training.last_eval_loss,
                    "mcc": self.training.last_eval_mcc,
                    **self.training.last_eval_confusion,
                },
            )
        except Exception as exc:  # noqa: BLE001
            self.events.add("error", "cola", "CoLA workflow failed", {"error": str(exc)})
            self.last_error = str(exc)
        finally:
            self.training.running = False


HTML_PAGE = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>TMT P2P Sync Node</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 1rem; background: #111; color: #f4f4f4; }
    h1, h2 { margin: 0.4rem 0; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
    .card { border: 1px solid #333; border-radius: 8px; padding: 0.8rem; background: #171717; }
    pre { white-space: pre-wrap; word-break: break-word; max-height: 240px; overflow-y: auto; }
    code { font-size: 0.9rem; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  </style>
</head>
<body>
  <h1>TMT Peer-to-Peer Model Sync</h1>
  <div class=\"grid\">
    <section class=\"card\"><h2>Node</h2><pre id=\"node\"></pre></section>
    <section class=\"card\"><h2>Training</h2><pre id=\"training\"></pre></section>
    <section class=\"card\"><h2>Synchronization</h2><pre id=\"sync\"></pre></section>
    <section class=\"card\"><h2>Sharing Policy</h2><pre id=\"policy\"></pre></section>
  </div>
  <section class=\"card\" style=\"margin-top:1rem;\"><h2>Activity Log</h2><pre id=\"events\"></pre></section>
<script>
const nodeEl = document.getElementById('node');
const trainingEl = document.getElementById('training');
const syncEl = document.getElementById('sync');
const policyEl = document.getElementById('policy');
const eventsEl = document.getElementById('events');

function pretty(obj) { return JSON.stringify(obj, null, 2); }

async function refresh() {
  const response = await fetch('/api/status');
  const data = await response.json();
  nodeEl.textContent = pretty(data.node);
  trainingEl.textContent = pretty(data.training);
  syncEl.textContent = pretty(data.sync);
  policyEl.textContent = pretty(data.sync.share_policy);
  eventsEl.textContent = pretty(data.events);
}

const eventSource = new EventSource('/api/events');
eventSource.onmessage = () => refresh();
setInterval(refresh, 2000);
refresh();
</script>
</body>
</html>
"""


def build_app(node: SyncNode) -> FastAPI:
    app = FastAPI(title="TMT P2P Sync")

    @app.on_event("startup")
    async def _startup() -> None:
        await node.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await node.shutdown()

    @app.get("/", response_class=HTMLResponse)
    async def home() -> str:
        return HTML_PAGE

    @app.get("/api/status")
    async def api_status() -> JSONResponse:
        return JSONResponse(node.status_payload())

    @app.get("/api/events")
    async def api_events() -> StreamingResponse:
        return StreamingResponse(node.events.stream(), media_type="text/event-stream")

    @app.get("/sync/manifest")
    async def sync_manifest(x_sync_token: str | None = Header(default=None)) -> JSONResponse:
        node._check_auth_header(x_sync_token)
        return JSONResponse(
            {
                "protocol_version": node.protocol_version,
                "architecture_id": node.manifest.architecture_id,
                "manifest_hash": node.manifest.manifest_hash,
                "tensor_specs": [
                    {"name": spec.name, "shape": list(spec.shape), "dtype": spec.dtype}
                    for spec in node.manifest.tensor_specs
                ],
            }
        )

    @app.get("/sync/state")
    async def sync_state(x_sync_token: str | None = Header(default=None)) -> JSONResponse:
        node._check_auth_header(x_sync_token)
        tensor_sizes = {name: len(data) for name, data in node.core.tensor_bytes.items()}
        return JSONResponse(
            {
                "tip_batch_id": node.core.tip_batch_id,
                "sequence": node.core.sequence,
                "root": node.core.root,
                "tensor_hashes": node.core.tensor_hashes,
                "tensor_sizes": tensor_sizes,
            }
        )

    @app.get("/sync/batch/{batch_id}")
    async def sync_batch(batch_id: str, x_sync_token: str | None = Header(default=None)) -> JSONResponse:
        node._check_auth_header(x_sync_token)
        batch = node.core.batches.get(batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail="batch not found")
        return JSONResponse(batch.as_dict())

    @app.get("/sync/tensor/{name:path}")
    async def sync_tensor(
        name: str,
        offset: int = Query(default=0, ge=0),
        length: int = Query(default=1 << 20, gt=0),
        x_sync_token: str | None = Header(default=None),
    ) -> JSONResponse:
        node._check_auth_header(x_sync_token)
        if name not in node.core.tensor_bytes:
            raise HTTPException(status_code=404, detail="tensor not found")
        if length > node.chunk_size:
            raise HTTPException(status_code=400, detail="requested chunk too large")
        chunk, chunk_hash = node.core.tensor_chunk(name, offset, length)
        return JSONResponse(
            {
                "name": name,
                "offset": offset,
                "length": len(chunk),
                "chunk_sha256": chunk_hash,
                "data_b64": base64.b64encode(chunk).decode("ascii"),
            }
        )

    @app.websocket("/sync/ws")
    async def sync_ws(websocket: WebSocket) -> None:
        node._check_auth_ws(websocket)
        await websocket.accept()
        peer = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"
        node.inbound_peers.add(peer)
        node.events.add("info", "peer", "Inbound websocket connected", {"peer": peer})
        try:
            while True:
                raw = await websocket.receive_text()
                if len(raw) > node.max_payload_bytes * 2:
                    await websocket.send_text(json.dumps({"type": "error", "error": "message too large"}))
                    continue
                msg = json.loads(raw)
                msg_type = msg.get("type")
                if msg_type == "hello":
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "tip",
                                "tip_batch_id": node.core.tip_batch_id,
                                "sequence": node.core.sequence,
                                "root": node.core.root,
                            }
                        )
                    )
                elif msg_type == "batch":
                    batch = SyncCore.load_batch(msg["batch"])
                    result = node.core.accept_remote_batch(batch)
                    if result.accepted:
                        node.model_adapter.apply_tensor_bytes(node.core.tensor_bytes, node.manifest)
                        node.remote_batch_count += 1
                        node.last_sync_time = time.time()
                        node.events.add("info", "sync", "Applied inbound remote batch", {"peer": peer, "batch_id": batch.batch_id})
                        await websocket.send_text(json.dumps({"type": "ack", "batch_id": batch.batch_id}))
                    else:
                        node.events.add("warning", "validation", "Rejected inbound batch", {"peer": peer, "reason": result.reason})
                        await websocket.send_text(json.dumps({"type": "error", "error": result.reason}))
                elif msg_type == "tip":
                    remote_sequence = int(msg.get("sequence", -1))
                    if remote_sequence > node.core.sequence:
                        node.events.add("info", "repair", "Inbound peer reported newer tip; waiting for outbound repair pull")
                else:
                    await websocket.send_text(json.dumps({"type": "error", "error": f"unknown message type {msg_type}"}))
        except WebSocketDisconnect:
            pass
        finally:
            node.inbound_peers.discard(peer)
            node.events.add("info", "peer", "Inbound websocket disconnected", {"peer": peer})

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TMT web UI + P2P sync + optional training/evaluation")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--peer", action="append", default=[], help="Peer websocket URL (repeat flag)")
    parser.add_argument("--checkpoint", default="smaller-4.5m.safetensors")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--temp", type=float, default=0.75)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--sync-interval", type=float, default=3.0)
    parser.add_argument("--sync-max-payload-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--sync-chunk-size", type=int, default=256 * 1024)
    parser.add_argument("--sync-token", default=None, help="Optional shared token for /sync endpoints")
    parser.add_argument("--train-text-path", default=None, help="Text file for background byte-training loop")
    parser.add_argument("--run-train-loop", action="store_true", help="Enable background training loop")
    parser.add_argument("--run-cola", action="store_true", help="Enable CoLA head train/eval workflow")
    parser.add_argument("--cola-train-path", default=None)
    parser.add_argument("--cola-eval-path", default=None)
    parser.add_argument("--cola-epochs", type=int, default=3)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    try:
        from main import Model
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Failed to import MLX model runtime ({exc}). Install MLX in a supported environment "
            "or run this on a machine with working MLX binaries."
        ) from exc

    model = Model(dim=args.dim, layers=args.layers, temp=args.temp, lr=args.lr)
    model.load(args.checkpoint)

    node = SyncNode(
        model=model,
        checkpoint_path=args.checkpoint,
        host=args.host,
        port=args.port,
        peers=args.peer,
        sync_interval=args.sync_interval,
        max_payload_bytes=args.sync_max_payload_bytes,
        chunk_size=args.sync_chunk_size,
        auth_token=args.sync_token,
        train_text_path=args.train_text_path,
        train_loop=args.run_train_loop,
        cola_train_path=args.cola_train_path,
        cola_eval_path=args.cola_eval_path,
        cola_epochs=args.cola_epochs,
        run_cola=args.run_cola,
    )

    app = build_app(node)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
