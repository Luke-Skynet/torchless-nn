# Torchless Neural Nets

Convolutional Neural Nets, Vision Transformers, and GPT's all implemented from scratch in Cuda Numpy (CuPy).
Without torch's tensor or autograd engine, this library hand derives all gradient calculations with reverse accumulation.
While this library is designed to run on an Nvidia GPU (with tensor cores), all cupy calls used are numpy compatible. To run everything on CPU, simply find/replace all instances of ```cupy.``` with ```np.``` and the code with still work.

## Attention

Attention supports both plain multi-head and grouped query (GQA), rotary and partial-rotary position embeddings, per-head QK normalization, sliding window attention, and Gemma's shared key/value projection. Calling `MultiHeadAttention` with default arguments keeps the original fused QKV path and parameter layout, so existing checkpoints still load.

Queries are processed in blocks of `chunk_size` (default: one block, identical to computing the whole sequence at once). Setting it bounds peak memory by the block size instead of the sequence length, and makes a sliding window an actual saving rather than just a mask — the scores outside the window are never computed instead of being computed and discarded.

## Inference Mode

```python
from utils import inference_mode

with inference_mode():
    model = gemma_gpt(**config)
```

Layers built inside the context allocate no gradient, moment or variance buffers, and each layer's cached activations are released as soon as the next layer has consumed them. For a Gemma 4 31B configuration that is 473.93 GiB down to 115.11 GiB.

## Modules

* **activations.py** - ReLU, GeLU Approx, SiLU (Swish), and Softmax activations.
* **layers.py** - Convolution, BatchNorm, MaxPool, AveragePool, Flatten, Dense, Dropout, and Transformer (LayerNorm, RMSNorm, Attention, Rotary Embeddings, Gated Feed Forward, Logit Softcap) layers.
* **gemma.py** - Gemma 4 style model assembly: grouped query attention, QK norm, sandwich norms, interleaved sliding window and global attention, and p-RoPE.
* **network.py** - Network framework class with Cross Entropy loss criterion and AdamW optimization.
* **transformer_adapters.py** - ViT image to tokens embedding, ViT MLP classification head, GPT embedding and GPT prediction layers.
* **test_gemma.py** - Finite difference gradient checks for every layer, runnable on CPU (```python test_gemma.py```).
* **utils** - Layer interface, Residual Layer wrapper, basic image augmentation functions, and tensor initializers to keep all parameters in FP32/TF32.