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

This study investigates the challenges of ambiguity faced by Visual-Language Models (VLMs) in understanding spatial semantics. Spatial cognition, influenced by cognitive psychology, spatial science, and cultural contexts, often assigns directionality to objects. For instance, while a car is inherently non-directional, human usage scenarios typically imbue it with an assumed orientation. In natural language, spatial relationship descriptions frequently omit explicit reference frame specifications, leading to semantic ambiguity. Existing VLMs, due to insufficient annotation of reference frames and object orientations in training data, often produce inconsistent responses. Consider an image where a car is positioned on the left side facing left and a man stands on the right side facing the viewer: an egocentric perspective describes the man as "to the right of the car," whereas an allocentric perspective interprets him as "behind the car," highlighting semantic discrepancies arising from different reference frames. Such ambiguities can lead to erroneous decisions when robots rely on natural language for navigation and manipulation. To address this problem, we propose a structured spatial representation method for identifying and annotating key spatial elements in images, including scene descriptions, reference objects and their orientations, target objects and their orientations, as well as reference frame types. Based on this representation, we constructed a dataset. By fine-tuning with QLoRA, these spatial elements were integrated into a pre-trained VLM. Experimental results demonstrate that our approach significantly outperforms state-of-the-art models in spatial orientation reasoning tasks, effectively enhancing the ability of VLMs to resolve spatial semantic ambiguities.

---

## 🏷️ Keywords

Visual-Language Models, Spatial Semantic Ambiguity, Reference Frame, Egocentric/Allocentric, Multimodal Reasoning

---

## 📄 Please view the [full paper](./AlloEgo_VLM_Paper.pdf)
