# TokenPPO
Token-Level Reinforcement Learning for Diffusion Model Generation
Introduction

This repository contains a minimal implementation of TokenPPO,
a reinforcement learning framework designed for token-level control in diffusion-based image generation.

TokenPPO reformulates the Markov Decision Process (MDP) in the token space, combining aesthetic scoring and human preference signals.
This enables fine-grained control of the diffusion process, enhancing both detail fidelity and overall aesthetic quality.

For full details of the method and experiments, see the paper:https://zenodo.org/records/16739433

Features
Token-Level PPO for diffusion optimization
Aesthetic reward design with hybrid signals
Plug-and-play with existing Stable Diffusion pipelines
Especially effective for anime-style generation
