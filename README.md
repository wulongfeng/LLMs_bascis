# LLMs_bascis

Topics related to Large Language Models (LLMs) and Transformers:

cheatsheet: https://github.com/afshinea/stanford-cme-295-transformers-large-language-models/blob/main/en/cheatsheet-transformers-large-language-models.pdf
Course link of Stanford's CME 295: [https://github.com/afshinea/stanford-cme-295-transformers-large-language-models?tab=readme-ov-file
](https://cme295.stanford.edu/)
Course link of Stanford's CS 336: https://stanford-cs336.github.io/spring2024/

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
* Decoding Strategies

## Memory and Computation Optimization
* Flash Attention
* KV Cache
* Gradient checkpointing
* Memory efficient optimizers
* Quantization (INT4/INT8)
* Pruning techniques

## Model Training and Fine-tuning
* LoRA (Low-Rank Adaptation)
* Reinforcement Learning Approaches
  + RLHF (Reinforcement Learning from Human Feedback)
  + PPO (Proximal Policy Optimization)
  + DPO (Direct Preference Optimization)

## Optimization Techniques

* Rejection Sampling
* Inference Acceleration Methods
* Model Compression Strategies
  
