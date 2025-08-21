# LLMs_bascis

Topics related to Large Language Models (LLMs) and Transformers:

* Cheatsheet: https://github.com/afshinea/stanford-cme-295-transformers-large-language-models/blob/main/en/cheatsheet-transformers-large-language-models.pdf
* Course link of Stanford's CME 295: [https://github.com/afshinea/stanford-cme-295-transformers-large-language-models?tab=readme-ov-file
](https://cme295.stanford.edu/)
* Course link of Stanford's CS 336: https://stanford-cs336.github.io/spring2024/

## Fundamental Architecture

* Transformer Architecture Overview
* Attention mechanisms (self-attention, cross-attention, Multi-head)
* Feed-forward networks (FFN)
* Generalized Query Attention (GQA)
* Mixture of Experts (MOE)
* Encoder vs Decoder architectures
* Model scaling (depth, width, vocabulary size)

## Training Fundamentals
* Loss functions
* Optimizer choices (AdamW, Lion)
* Learning rate scheduling
* Gradient clipping
* Mixed precision training
* Distributed training strategies

## Core Components

* Tokenization Methods
* Position Embeddings
  + Traditional Position Embedding (PE)
  + Rotary Position Embedding (RoPE)
* Layer Normalization
* Decoding Strategies.
  + Blog: [How to generate](https://huggingface.co/blog/how-to-generate)
  + [Code](https://github.com/huggingface/transformers/blob/c4d4e8bdbd25d9463d41de6398940329c89b7fb6/src/transformers/generation_utils.py#L101) and [simpler version](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py)
  + Constraint Generation with [prefix_allowed_tokens_fn](https://discuss.huggingface.co/t/example-of-prefix-allowed-tokens-fn-while-text-generation/6635) and [Trie tree](https://github.com/facebookresearch/GENRE/blob/main/genre/trie.py)

## Memory and Computation Optimization
* Flash Attention
* KV Cache
* Gradient checkpointing
* Memory efficient optimizers
* Quantization (INT4/INT8)
* Pruning techniques
* Speculative decoding
   + [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/pdf/2211.17192)
   + [Tutorial](https://github.com/hemingkx/SpeculativeDecodingPapers)
   + [Blog](https://zhuanlan.zhihu.com/p/651359908)
  
## Model Training and Fine-tuning
* LoRA (Low-Rank Adaptation)
* Reinforcement Learning Approaches
  + RLHF (Reinforcement Learning from Human Feedback)
  + PPO (Proximal Policy Optimization). [Code](https://github.com/deepspeedai/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/rlhf/ppo_trainer.py)
  + DPO (Direct Preference Optimization)
    + [Code](https://github.com/TechxGenus/LightDPO/tree/main)
    +   [Data Construction](https://blog.gopenai.com/build-a-high-quality-dpo-dataset-596a07e574ec) 

## Optimization Techniques

* Rejection Sampling
* Inference Acceleration Methods
* Model Compression Strategies

## Other Resources
[R1](https://github.com/huggingface/open-r1)
  
