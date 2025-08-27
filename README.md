# Spatial Semantics for Visual-Language Models (VLMs)

This repository contains the implementation and dataset for enhancing **Visual-Language Models (VLMs)** in resolving spatial semantic ambiguities. Our work addresses the challenges VLMs face when interpreting spatial relationships in natural language without explicit reference frames.

## Overview

Understanding spatial semantics is inherently challenging because spatial cognition is influenced by **cognitive psychology**, **spatial science**, and **cultural contexts**. Objects in a scene often carry an implied directionality. For instance, a car may be non-directional by nature, but human usage scenarios typically assign it an orientation.  

Natural language descriptions often **omit explicit reference frames**, leading to semantic ambiguity. For example, in an image where:

- A car is on the left side, facing left.
- A man is on the right side, facing the viewer.

Different perspectives yield different descriptions:

- **Egocentric perspective**: "the man is to the right of the car."
- **Allocentric perspective**: "the man is behind the car."

Such ambiguities can cause **erroneous decisions** in robotics tasks that rely on natural language for navigation or manipulation.

## Our Approach

We propose a **structured spatial representation** method to identify and annotate key spatial elements in images:

- **Scene descriptions**
- **Reference objects and their orientations**
- **Target objects and their orientations**
- **Reference frame types** (egocentric/allocentric)

Based on this representation, we constructed a **spatially annotated dataset** and fine-tuned a pre-trained VLM using **QLoRA**, integrating these spatial elements into the model.

## Results

Experimental results show that our method:

- Significantly outperforms state-of-the-art models in **spatial orientation reasoning tasks**
- Effectively enhances VLMs’ ability to resolve **spatial semantic ambiguities**

## Keywords

Visual-Language Models, Spatial Semantic Ambiguity, Reference Frame, Egocentric/Allocentric, Multimodal Reasoning

