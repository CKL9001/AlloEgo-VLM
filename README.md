# 🚀 AlloEgo-VLM: Resolving Allocentric and Egocentric Orientation Ambiguities in Visual-Language Model(s)

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

## 🏷️ Keywords

Visual-Language Models, Spatial Semantic Ambiguity, Reference Frame, Egocentric/Allocentric, Multimodal Reasoning

---
