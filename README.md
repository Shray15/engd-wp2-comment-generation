# WP2: Synthetic Comment Generation for Dutch Housing Associations

## Overview

This repository contains the code and experiments for **Work Package 2 (WP2)** of an Engineering Doctorate (EngD) research project at **Eindhoven University of Technology (TU/e)**, in collaboration with **Woonbedrijf**.

The goal is to generate human-like synthetic Dutch Facebook comments in response to housing association posts, using fine-tuned large language models. Generated comments are evaluated against a hand-labelled test set on three qualitative authenticity dimensions: sentiment, intent, and relatedness to the post.

---

## Models

Three instruction-tuned language models are the primary focus:

| Model | Type |
|---|---|
| `Llama-3.2-3B-Instruct` | Instruction-tuned |
| `Gemma-4B-IT` | Instruction-tuned |
| `Fietje-2.7-IT` | Instruction-tuned (Dutch) |

> **Experimental:** `Qwen-4B` (base chat model) is additionally instruction fine-tuned as an exploratory baseline outside the main comparison.

---

## Approaches

### 1 · Few-Shot + Length Distribution *(Baseline)*
No fine-tuning. A Dutch tenant persona is prompted with 3 real comment examples, the post, and a word-count range sampled from the empirical length distribution of the training set. Establishes the upper bound achievable without parameter updates.

### 2 · LoRA Fine-Tuning + Objective Conditioning + Length
LoRA fine-tuning with CLM loss. The system prompt explicitly states three evaluation criteria (sentiment, intent, relatedness). A 4-category length label derived from real comment word counts is injected at train and inference time:
`zeer kort ≤5` · `kort ≤12` · `gemiddeld ≤25` · `lang >25`

### 3 · LoRA Fine-Tuning + Multi-Head Loss
Identical setup to Approach 2, but with auxiliary loss heads added at training time:

$$\mathcal{L}_{total} = \mathcal{L}_{CLM} + \lambda_s \mathcal{L}_{sent} + \lambda_i \mathcal{L}_{intent} + \lambda_r \mathcal{L}_{rel}$$

where $\mathcal{L}_{sent}$, $\mathcal{L}_{intent}$ are cross-entropy and $\mathcal{L}_{rel}$ is MSE. Auxiliary heads are dropped at inference time. Tests whether gradient-level supervision on authenticity dimensions outperforms prompt conditioning alone.

---

## Evaluation

All approaches evaluated on a held-out test set of **208 hand-labelled comments** across three dimensions:

| Dimension | Metric |
|---|---|
| Sentiment | JS Divergence |
| Intent | JS Divergence |
| Relatedness | JS Divergence |

Target threshold: JS Distance < 0.15.

---

## Data Split

| Total | Train | Val | Test |
|---|---|---|---|
| 3,594 | 2,875 | 719 | 208 |

---

## Repository Structure
```
WP2_Simulation_Tool/
│
├── Few Shot Prompt/
│   ├── llama/                 # Few-shot generation code for Llama
│   ├── gemma/                 # Few-shot generation code for Gemma
│   └── fietje/                # Few-shot generation code for Fietje
│
├── Fine_Tune_LLAMA/
│   ├── code/                  # Fine-tuning + generation scripts
│   ├── objective/             # Results: sentiment, intent, relatedness plots (Approach 2)
│   └── multiloss/             # Results: sentiment, intent, relatedness plots (Approach 3)
│
├── Fine Tune gemma/
│   ├── code/
│   ├── objective/
│   └── multiloss/
│
├── Fine_tune_fietje/
│   ├── code/
│   ├── objective/
│   └── multiloss/
│
├── Fine_tune_qwen/            # Experimental: Qwen-4B base chat model, instruction fine-tuned
│
├── Generation and Evaluation/ # Shared generation + evaluation pipeline
├── analysis/                  # JS Divergence tables and cross-model comparison outputs
├── data/                      # Raw and processed data (excluded from repo)
├── templates/                 # Prompt templates
├── utils/                     # Shared utility functions
│
├── .gitignore
└── README.md
```

---

## Setup
```bash
pip install -r requirements.txt
```

Create a `.env` file with your HuggingFace token:
```
HF_TOKEN=your_token_here
```

---

## Context

This work is part of an EngD thesis in the **Built Environment** department at TU/e, applying NLP to tenant engagement and energy transition communication in Dutch social housing.