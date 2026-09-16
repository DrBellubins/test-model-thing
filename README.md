# Test-Model-Thing (TMT)

[YouTube Video](https://youtu.be/9UERVVwpNew)

This is a small proof-of-concept language model (not an LLM) that incorporates the following (and some smaller features as well):
* Latent-space prediction
* Internal state + recurrent trace units (RTUs)
* Byte input/output
* Continuous data streaming
* Test-time training

The model is built with MLX, so it should run fine on all Apple Silicon devices. MLX on Linux has not been tested, but feel free to try it.

Being a proof of concept I have only trained a 4.5-million parameter model (keep in mind, GPT-1 was ~117m) for about 12 hours, but there are very promising results. The model tends to misspell characters (since it outputs byte-by-byte, rather than token-by-token) but it is able to close quotes/brackets and such. Given further training and scaling up the hyperparameters this could become much more powerful. My dataset is also tiny (only a few hundred MB), so there's a lot more world knowledge that can be fed into the model.

This model architecture was designed in about a month by me (a solo high school dev) and some Gemini (only pair programming, no agents). I wrote about a dozen prototypes before creating this architecture. I write READMEs myself without AI.

Feel free to fork the training and benchmark code (everything is under MIT). I really encourage you to try things out, submit issues, and fork the repo.

<img width="499" height="497" alt="3f7f1530-c0c7-43c4-9981-30e9023a19fb" src="https://github.com/user-attachments/assets/eb7e5a97-09b5-4a7b-9484-eb898042e9dc" />

## Peer-to-peer sync web application (single entry point)

This repository now includes `serve.py`, a master script that runs:
- a browser UI
- a local P2P synchronization node (websocket + HTTP endpoints)
- optional background workflows (byte-training loop and/or CoLA head train/eval)

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Start one node

```bash
python serve.py \
  --host 127.0.0.1 \
  --port 8080 \
  --checkpoint smaller-4.5m.safetensors \
  --dim 512 --layers 16 --temp 0.75 --lr 5e-4
```

Open `http://127.0.0.1:8080`.

### Start two local peers

Terminal 1:
```bash
python serve.py --port 8080 --checkpoint node1.safetensors --peer ws://127.0.0.1:8081/sync/ws
```

Terminal 2:
```bash
python serve.py --port 8081 --checkpoint node2.safetensors --peer ws://127.0.0.1:8080/sync/ws
```

Optional shared-token auth for sync endpoints/peer websockets:
```bash
python serve.py --port 8080 --sync-token my-shared-token
python serve.py --port 8081 --sync-token my-shared-token --peer ws://127.0.0.1:8080/sync/ws
```

### Optional background training/evaluation

Run local byte-training loop:
```bash
python serve.py --run-train-loop --train-text-path /absolute/path/to/training.txt
```

Run CoLA head training/eval only when paths are provided:
```bash
python serve.py \
  --run-cola \
  --cola-train-path /absolute/path/to/CoLA/original/raw/in_domain_train.tsv \
  --cola-eval-path /absolute/path/to/CoLA/original/raw/in_domain_dev.tsv \
  --cola-epochs 3
```

CoLA files are **not required** unless `--run-cola` is enabled.

### Synchronization design (local-first, deterministic, no blockchain claims)

- All peers must use matching trainable tensor manifests (same tensor names/shapes/dtypes).
- Manifest identity is deterministic and architecture-based.
- Runtime state comparison uses per-tensor SHA-256 hashes and a canonical aggregate model root.
- Normal sync transmits versioned update batches containing only changed trainable tensors.
- Each incoming batch is validated before apply:
  - protocol version
  - architecture/manifest identity
  - predecessor/tip + sequence relation
  - tensor metadata (name/shape/dtype)
  - payload length/hash + batch-id hash
  - resulting model root
- If predecessor/order diverges, nodes reject the batch and perform deterministic snapshot repair.
- Snapshot repair uses chunked, hash-verified tensor transfer over HTTP.
- Transient recurrent traces (`embedtrace`, `state.*`, `decaytrace.*`) and optimizer state (`o.*`) are not consensus-shared state.

Conflict strategy: this is a trusted cooperative swarm strategy (not PoW/PoS). Nodes accept only direct-tip successor batches. Fork/out-of-order batches are rejected, logged, and reconciled by snapshot repair from a healthy peer.

Trust/security limits: SHA-256 protects integrity and accidental corruption. Optional shared-token auth reduces accidental cross-talk, but this is **not** a hostile-network Byzantine consensus system.

## Training your own model

Model weights (in ```.safetensors```) are not provided because GitHub doesn't like very large files. But, you can train your own model simply by initializing a ```venv``` and installing ```mlx```, no other libraries needed, then running ```main.py```. When you run it, you will be prompted with the mode, ```train``` being train on dataset and ```chat``` being chat. There is also ```chatreadonly``` for readonly chat (the model weights will not re-save to disk and override things) and ```chatnotrace``` if you want to break things. You will have to configure your own dataset by modifying the code (to run dataset mode), but you should be able to run chat mode without modifying anything if you have weights already.

Once it begins training, you can safely ^C the program and it will save weights. It should also periodically save weights if I'm not mistaken. The saved weights include the internal memory so the model will remember that the next time it runs. You can launch into chat mode and the memory should carry on from whatever it was learning in training.

## How it works

In detail, here are some of the main capabilities of the model that differ from LLMs:
* JEPA-style latent space prediction, as the decoder can be removed/disabled and the model still rolls out forward as is. The model is not trained explicitly on predicting the next byte, but rather on two separate goals (predicting the next 'thing' in latent space, and translating the current latent space vector to a byte).
* Theoretically infinite memory, as it does not have a context window and instead relies on RTUs to store internal state/memory. However it does decay old memories over time. Also I think this should be O(1) memory based on my implementation but I'm not 100% sure.
* Built-in multimodality, as the model outputs bytes (and thus should theoretically be capable of handling any binary data).
* Streaming data live, since the model only processes one byte at once at rapid pace. In fact it is completely 'blind' to everything that came before the current byte, only relying on the current processed byte and its internal memory to decide the next byte. This confirms the model is definitely learning to remember things.
* Continual training, as it keeps training on user input, training data, and its own output to improve its predictions automatically. Keep in mind the model can only output a byte (0-255) each pass anyways (in addition to updating its own state).

The two important hyperparameters are the size of the latent vector (dim) and the amount of individual state layers the latent passes through before decoding (layers). For my 4.5m test these are ```dim = 512``` and ```layers = 16```. There are some other configurations you can change but I think they are less important.

I think this probably will contribute significantly to solving continual learning and memory but I still need other people to review and verify my work! Please feel free to open GitHub issues to tell me what's wrong. If you have compute (e.g. you are a lab or just have GPUs lying around), feel free to fork my code and train larger models as well, with credit. I personally don't have enough compute and as such I can't really train very large models.

Below is an approximate flow chart of the model architecture, made in Apple's Freeform app (excluding the wrapper for dataset cleaning and input/output handling) for reference. Note that the arrow connecting the target latent to the CE loss should instead be the target byte to the CE loss.

<img width="1653" height="1161" alt="JEPA thing" src="https://github.com/user-attachments/assets/2d3a34ff-ba6a-44b8-b361-6c73da9216c0" />
