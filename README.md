# My Torch Code Solutions

Personal solution and note for implementing common ML and DL operators, following the exercises from [duoan/TorchCode](https://github.com/duoan/TorchCode).

# 📋 Problem Set

> **Frequency**: 🔥 = very likely in interviews, ⭐ = commonly asked, 💡 = emerging / differentiator

## 🧱 Fundamentals — "Implement X from scratch"

The bread and butter of ML coding interviews. You'll be asked to write these without `torch.nn`.

|  #  | Problem               | What You'll Implement                 |                               Difficulty                                | Freq | Key Concepts                                      |
| :-: | --------------------- | ------------------------------------- | :---------------------------------------------------------------------: | :--: | ------------------------------------------------- |
|  1  | ReLU                  | `relu(x)`                             |   ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square)   |  🔥  | Activation functions, element-wise ops            |
|  2  | Softmax               | `my_softmax(x, dim)`                  |   ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square)   |  🔥  | Numerical stability, exp/log tricks               |
| 16  | Cross-Entropy Loss    | `cross_entropy_loss(logits, targets)` |   ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square)   |  🔥  | Log-softmax, logsumexp trick                      |
| 17  | Dropout               | `MyDropout` (nn.Module)               |   ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square)   |  🔥  | Train/eval mode, inverted scaling                 |
| 18  | Embedding             | `MyEmbedding` (nn.Module)             |   ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square)   |  🔥  | Lookup table, `weight[indices]`                   |
| 19  | GELU                  | `my_gelu(x)`                          |   ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square)   |  ⭐  | Gaussian error linear unit, `torch.erf`           |
| 20  | Kaiming Init          | `kaiming_init(weight)`                |   ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square)   |  ⭐  | `std = sqrt(2/fan_in)`, variance scaling          |
| 21  | Gradient Clipping     | `clip_grad_norm(params, max_norm)`    |   ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square)   |  ⭐  | Norm-based clipping, direction preservation       |
| 31  | Gradient Accumulation | `accumulated_step(model, opt, ...)`   |   ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square)   |  💡  | Micro-batching, loss scaling                      |
| 40  | Linear Regression     | `LinearRegression` (3 methods)        | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) |  🔥  | Normal equation, GD from scratch, nn.Linear       |
|  3  | Linear Layer          | `SimpleLinear` (nn.Module)            | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) |  🔥  | `y = xW^T + b`, Kaiming init, `nn.Parameter`      |
|  4  | LayerNorm             | `my_layer_norm(x, γ, β)`              | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) |  🔥  | Normalization, running stats, affine transform    |
|  7  | BatchNorm             | `my_batch_norm(x, γ, β)`              | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) |  ⭐  | Batch vs layer statistics, train/eval behavior    |
|  8  | RMSNorm               | `rms_norm(x, weight)`                 | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) |  ⭐  | LLaMA-style norm, simpler than LayerNorm          |
| 15  | SwiGLU MLP            | `SwiGLUMLP` (nn.Module)               | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) |  ⭐  | Gated FFN, `SiLU(gate) * up`, LLaMA/Mistral-style |
| 22  | Conv2d                | `my_conv2d(x, weight, ...)`           | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) |  🔥  | Convolution, unfold, stride/padding               |

## 🧠 Attention Mechanisms — The heart of modern ML interviews

If you're interviewing for any role touching LLMs or Transformers, expect at least one of these.

|  #  | Problem                      | What You'll Implement                   |                               Difficulty                                | Freq | Key Concepts                                              |
| :-: | ---------------------------- | --------------------------------------- | :---------------------------------------------------------------------: | :--: | --------------------------------------------------------- |
| 23  | Cross-Attention              | `MultiHeadCrossAttention` (nn.Module)   | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) |  ⭐  | Encoder-decoder, Q from decoder, K/V from encoder         |
|  5  | Scaled Dot-Product Attention | `scaled_dot_product_attention(Q, K, V)` |   ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square)   |  🔥  | `softmax(QK^T/√d_k)V`, the foundation of everything       |
|  6  | Multi-Head Attention         | `MultiHeadAttention` (nn.Module)        |   ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square)   |  🔥  | Parallel heads, split/concat, projection matrices         |
|  9  | Causal Self-Attention        | `causal_attention(Q, K, V)`             |   ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square)   |  🔥  | Autoregressive masking with `-inf`, GPT-style             |
| 10  | Grouped Query Attention      | `GroupQueryAttention` (nn.Module)       |   ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square)   |  ⭐  | GQA (LLaMA 2), KV sharing across heads                    |
| 11  | Sliding Window Attention     | `sliding_window_attention(Q, K, V, w)`  |   ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square)   |  ⭐  | Mistral-style local attention, O(n·w) complexity          |
| 12  | Linear Attention             | `linear_attention(Q, K, V)`             |   ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square)   |  💡  | Kernel trick, `φ(Q)(φ(K)^TV)`, O(n·d²)                    |
| 14  | KV Cache Attention           | `KVCacheAttention` (nn.Module)          |   ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square)   |  🔥  | Incremental decoding, cache K/V, prefill vs decode        |
| 24  | RoPE                         | `apply_rope(q, k)`                      |   ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square)   |  🔥  | Rotary position embedding, relative position via rotation |
| 25  | Flash Attention              | `flash_attention(Q, K, V, block_size)`  |   ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square)   |  💡  | Tiled attention, online softmax, memory-efficient         |

## 🏗️ Architecture & Adaptation — Put it all together

|  #  | Problem             | What You'll Implement          |                               Difficulty                                | Freq | Key Concepts                                                |
| :-: | ------------------- | ------------------------------ | :---------------------------------------------------------------------: | :--: | ----------------------------------------------------------- |
| 26  | LoRA                | `LoRALinear` (nn.Module)       | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) |  ⭐  | Low-rank adaptation, frozen base + `BA` update              |
| 27  | ViT Patch Embedding | `PatchEmbedding` (nn.Module)   | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) |  💡  | Image → patches → linear projection                         |
| 13  | GPT-2 Block         | `GPT2Block` (nn.Module)        |   ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square)   |  ⭐  | Pre-norm, causal MHA + MLP (4x, GELU), residual connections |
| 28  | Mixture of Experts  | `MixtureOfExperts` (nn.Module) |   ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square)   |  ⭐  | Mixtral-style, top-k routing, expert MLPs                   |

## ⚙️ Training & Optimization

|  #  | Problem             | What You'll Implement           |                               Difficulty                                | Freq | Key Concepts                        |
| :-: | ------------------- | ------------------------------- | :---------------------------------------------------------------------: | :--: | ----------------------------------- |
| 29  | Adam Optimizer      | `MyAdam`                        | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) |  ⭐  | Momentum + RMSProp, bias correction |
| 30  | Cosine LR Scheduler | `cosine_lr_schedule(step, ...)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) |  ⭐  | Linear warmup + cosine annealing    |

## 🎯 Inference & Decoding

|  #  | Problem                | What You'll Implement                    |                               Difficulty                                | Freq | Key Concepts                                |
| :-: | ---------------------- | ---------------------------------------- | :---------------------------------------------------------------------: | :--: | ------------------------------------------- |
| 32  | Top-k / Top-p Sampling | `sample_top_k_top_p(logits, ...)`        | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) |  🔥  | Nucleus sampling, temperature scaling       |
| 33  | Beam Search            | `beam_search(log_prob_fn, ...)`          | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) |  🔥  | Hypothesis expansion, pruning, eos handling |
| 34  | Speculative Decoding   | `speculative_decode(target, draft, ...)` |   ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square)   |  💡  | Accept/reject, draft model acceleration     |

## 🔬 Advanced — Differentiators

|  #  | Problem           | What You'll Implement                                    |                             Difficulty                              | Freq | Key Concepts                                                                  |
| :-: | ----------------- | -------------------------------------------------------- | :-----------------------------------------------------------------: | :--: | ----------------------------------------------------------------------------- |
| 35  | BPE Tokenizer     | `SimpleBPE`                                              | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) |  💡  | Byte-pair encoding, merge rules, subword splits                               |
| 36  | INT8 Quantization | `Int8Linear` (nn.Module)                                 | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) |  💡  | Per-channel quantize, scale/zero-point, buffer vs param                       |
| 37  | DPO Loss          | `dpo_loss(chosen, rejected, ...)`                        | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) |  💡  | Direct preference optimization, alignment training                            |
| 38  | GRPO Loss         | `grpo_loss(logps, rewards, group_ids, eps)`              | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) |  💡  | Group relative policy optimization, RLAIF, within-group normalized advantages |
| 39  | PPO Loss          | `ppo_loss(new_logps, old_logps, advantages, clip_ratio)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) |  💡  | PPO clipped surrogate loss, policy gradient, trust region                     |

## Supplements

| Category | Topic                  | Name                                              | File                                               |
| -------- | ---------------------- | ------------------------------------------------- | -------------------------------------------------- |
| Misc     | \                      | Dataset, Model, Training and Testing with PyTorch | [train_test.ipynb](supplements/train_test.ipynb)   |
| Module   | Vision: Segmentation   | Box IOU                                           | [box_iou.ipynb](supplements/box_iou.ipynb)         |
| Module   | Vision                 | Convolution                                       | [convolution.ipynb](supplements/convolution.ipynb) |
| Model    | Vision: Classification | AlexNet                                           | [alexnet.ipynb](supplements/alexnet.ipynb)         |
| Model    | Vision: Classification | ResNet                                            | [resnet.ipynb](supplements/resnet.ipynb)           |

TODO:

- Classification
    - VGG
    - Inception V1: Inception Module
    - \*ResNet
    - Xception: Depthwise Separable Convolution
    - DenseNet: Dense Block
    - \*MobileNet V1: Depthwise Separable Convolution
    - MobileNet V2: Inverted Residual Block
    - \*ViT
- Opject Detection
    - 2 Stage
        - Fast R-CNN: ROI Pooling
        - \*Faster R-CNN: Region Proposal Network
    - 1 Stage
        - YOLO v1
        - SSD
        - RetinaNet: Focal Loss
        - \*YOLO v3
        - YOLO Anchor-free
        - FCOS: Center-ness
        - \*DETR
- Image Segmentation
    - Semantic Segmentation
        - FCN: Deconvolution
        - \*U-Net: Skip Connection
        - \*DeepLab V3+: ASPP, Atrous Convolution
    - Instance Segmentation
        - Mask R-CNN
    - Panoptic Segmentation
        - Mask 2 Former
        - SAM: Prompt Encoder, Mask Decoder
- Time Series
    - RNN
    - LSTM
    - GRU
    - Transformer
    - \*Swin Transformer
- NLP
- Generative Model
    - \*DDPM
    - VAE
    - GAN
    - \*DCGAN
- Multimodal
    - \*CLIP
    - MAE
- ConvNeXt
