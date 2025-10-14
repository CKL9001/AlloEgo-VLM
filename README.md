# 🚀 AlloEgo-VLM: Disambiguating Allocentric and Egocentric Reference Frames in Vision–Language Models

This repository contains the implementation and dataset for enhancing **Visual-Language Models (VLMs)** in resolving **spatial semantic ambiguities**. Our work tackles the challenges VLMs face when interpreting spatial relationships in natural language without explicit reference frames.

---

## 🌐 Overview

Understanding spatial semantics is challenging because **spatial cognition** is influenced by:

- 🧠 Cognitive psychology  
- 📏 Spatial science  
- 🌏 Cultural contexts  

Objects often carry an **implied directionality**. For example, a car may be non-directional, but humans typically assign it an orientation in real-world scenarios.  

Natural language descriptions often **omit explicit reference frames**, causing **semantic ambiguity**.  

Example:  
<p align="center">
  <img src="car-man.png" alt="Car and Man Spatial Ambiguity" width="400"/>
</p>

- 🚗 Car on the left side, facing left  
- 🧍 Man on the right side, facing the viewer  

Different perspectives produce different descriptions:

- 👁️ **Egocentric perspective**: "the man is to the right of the car"  
- 🌍 **Allocentric perspective**: "the man is behind the car"  

Such ambiguities can lead to **wrong decisions in robotics** relying on natural language for navigation or manipulation.

---

## 🛠️ Our Approach

We propose a **structured spatial representation** method to identify and annotate key spatial elements in images:

- 🖼️ **Scene descriptions**  
- 🏷️ **Reference objects & orientations**  
- 🎯 **Target objects & orientations**  
- 🔄 **Reference frame types** (egocentric/allocentric)  

Based on this representation, we constructed a **spatially annotated dataset** and fine-tuned a pre-trained VLM using **QLoRA**, integrating these spatial elements into the model.

---

## 📊 Results

Our method:

- ⭐ **Outperforms state-of-the-art models** in spatial orientation reasoning tasks  
- 🤖 **Enhances VLMs’ ability to resolve spatial semantic ambiguities**

---

## 📄 Abstract

This study investigates the challenge of ambiguity faced by Vision--Language Models (VLMs) in understanding spatial semantics. Spatial cognition, shaped by cognitive psychology, spatial science, and cultural context, often assigns directionality to objects. However, natural language descriptions of spatial relations frequently omit explicit reference frames, leading to semantic ambiguity and potentially serious errors for embodied AI robots. Existing VLMs, due to insufficient training on reference frames and object orientations, often produce inconsistent responses. To address this issue, we construct a new dataset, _AlloEgo-View_, comprising (image, query, view-specific answer) triplets that capture key object relations from both allocentric and egocentric perspectives. The view-specific descriptions follow a structured spatial representation that annotate key elements, including detailed scene descriptions, reference and target objects, their orientations, reference frames, and view types. Building on AlloEgo-View, we develop _AlloEgo-VLM_, a framework designed to disambiguate allocentric and egocentric reference frames, even under ambiguous queries, and to be easily integrated into existing VLMs via supervised fine-tuning. Experiments highlight the limitations of current VLMs in handling view-specific queries and demonstrate the strong disambiguation ability of AlloEgo-VLM.

---

## 🏷️ Keywords

Vision-Language Models, Spatial Ambiguity, Reference Frames, Egocentric/Allocentric, Multimodal Reasoning

---

## 📄 Please view the [Full_Paper](./AlloEgo_VLM_Paper.pdf)
