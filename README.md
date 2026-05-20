# RLHF with PPO: SmolLM2 Tutorial

A step-by-step tutorial on implementing Reinforcement Learning from Human Feedback (RLHF) with Proximal Policy Optimization (PPO) using SmolLM2 in PyTorch.

##  Overview
This repository contains a Google Colab notebook demonstrating how to align a Large Language Model (LLM) to human preferences. It breaks down the complex RLHF pipeline into three manageable phases using the lightweight **SmolLM2-135M** model, making it possible to train and experiment on standard GPUs.

### The pipeline consists of 3 main steps:
1. **Supervised Fine-Tuning (SFT):** Training the base model to imitate high-quality conversational responses using standard cross-entropy loss.
2. **Reward Modeling (RM):** Modifying the LLM into a Critic/Reward model that outputs a scalar score to evaluate the quality of a response based on human preference data (Bradley-Terry model).
3. **Reinforcement Learning (PPO):** Using Proximal Policy Optimization to train the SFT model (Actor) to maximize the scores given by the Reward Model, while using a Value Model (Critic) and KL divergence penalties to prevent catastrophic forgetting and over-optimization.

##  Models and Datasets
* **Base Model:** [`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
* **Dataset:** [`HumanLLMs/Human-Like-DPO-Dataset`](https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset) (Annotated for natural, informal, and conversational human interaction).

##  Key Concepts Covered
* Formatting datasets into **ChatML** format.
* Left vs. Right padding strategies for sequence generation and reward scoring.
* Generalized Advantage Estimation (GAE) for stable reinforcement learning updates.
* Clipped surrogate objective functions for Policy and Value networks.

##  Acknowledgments
* The theoretical framework and notebook structure are inspired by the original [InstructGPT / RLHF paper (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) and [PPO paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347).
