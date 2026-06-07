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

## 📂 AlloEgo-View Dataset

We constructed a new dataset, **AlloEgo-View**, comprising (image, query, view-specific answer) triplets. Due to license restrictions of the original datasets and to ensure the reproducibility of our specific preprocessing (cropping, resizing, and spatial annotation), we host the **processed subset** of images on Google Drive.

### 📥 Download
You can download the processed dataset here:
👉 **[Download AlloEgo-View Dataset (Google Drive)](https://drive.google.com/drive/folders/1YT_u0dPRujq1dPUOMwUKsuxCqrcAvfP0?usp=sharing)**

### ⚖️ License & Acknowledgement
The images in **AlloEgo-View** are derived and processed from the following datasets. We do not own the copyright of the original images; they belong to their respective creators and datasets.

* **[COCO Dataset](https://cocodataset.org/#home)**
* **[GQA Dataset](https://cs.stanford.edu/people/dorarad/gqa/index.html)**
* **[SPAR Dataset](https://logosroboticsgroup.github.io/SPAR/)**
* **[NYU Depth Dataset v2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)**


> **Disclaimer:** This processed dataset is distributed for **academic and research purposes only**. If you use this dataset, please consider citing the original works listed above in addition to our paper.

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

This study investigates the challenge of ambiguity faced by Vision-Language Models (VLMs) in understanding spatial semantics.  
Spatial cognition, shaped by cognitive psychology, spatial science, and cultural context, often assigns directionality to objects.  
However, natural language descriptions of spatial relations frequently omit explicit reference frames, leading to semantic ambiguity and potentially serious errors for embodied AI robots.  
Existing VLMs, due to insufficient training on reference frames and object orientations, often produce inconsistent responses.  
To address this issue, we construct a new dataset, AlloEgo-View, comprising (image, query, view-specific answer) triplets that capture key object relations from both allocentric and egocentric perspectives.
The view-specific descriptions follow a structured spatial representation that annotate detailed scene descriptions, reference and target objects, their orientations, reference frames, and view types. 
Building on AlloEgo-View, we develop AlloEgo-VLM, a framework to disambiguate allocentric and egocentric reference frames, even under ambiguous queries, and to be easily integrated into existing VLMs via supervised fine-tuning.  
Furthermore, we deploy our framework onto an embodied robotic platform within NVIDIA Isaac Sim to validate its real-world feasibility in open-ended object searching tasks.
Experiments highlight the limitations of current VLMs in handling view-specific queries and demonstrate the strong disambiguation ability of AlloEgo-VLM.

---

## 🏷️ Keywords

Vision-Language Models, Spatial Ambiguity, Reference Frames, Egocentric/Allocentric, Multimodal Reasoning

---

## 📄 Please view the [Full_Paper](./AlloEgo_VLM_Paper.pdf)
