# Torchless Neural Nets

Convolutional Neural Nets, Vision Transformers, and GPT's all implemented from scratch in Cuda Numpy (CuPy).
Without torch's tensor or autograd engine, this library hand derives all gradient calculations with reverse accumulation.
While this library is designed to run on an Nvidia GPU (with tensor cores), all cupy calls used are numpy compatible. To run everything on CPU, simply replace all instances of ```import cupy``` with ```import numpy as cupy``` and the code with still work.

## Modules

* **activations.py** - ReLU, GeLU Approx, SiLU (Swish), and Softmax activations.
* **layers.py** - Convolution, BatchNorm, MaxPool, AveragePool, Flatten, Dense, Dropout, and Transformer (LayerNorm, Attention, Feed Forward) layers.
* **network.py** - Network framework class with Cross Entropy loss criterion and AdamW optimization.
* **transformer_adapters.py** - ViT image to tokens embedding, ViT MLP classification head, GPT embedding and GPT prediction layers.
* **utils** - Layer interface, Residual Layer wrapper, and tensor initializers to keep all parameters in FP32/TF32.