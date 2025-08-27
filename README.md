🚀 AlloEgo-VLM: Resolving Allocentric and Egocentric Orientation Ambiguities in Visual-Language Models
<p align="center"> <img src="car-man.png" alt="Car and Man Spatial Ambiguity" width="450"/> </p>

This repository provides the implementation, dataset, and methodology for enhancing Visual-Language Models (VLMs) in resolving spatial semantic ambiguities caused by missing or implicit reference frames.

🌐 Overview

Understanding spatial semantics in language–vision tasks is difficult because human spatial cognition is influenced by:

🧠 Cognitive psychology

📏 Spatial science

🌏 Cultural contexts

Objects often carry implied directionality.
For example, a car is inherently non-directional, yet humans typically assign it a forward-facing orientation in real-world use.

⚠️ The Problem: Ambiguity in Reference Frames

Natural language descriptions often omit explicit reference frames, causing semantic ambiguity.

🚗 Car on the left side, facing left

🧍 Man on the right side, facing the viewer

Different interpretations arise:

👁️ Egocentric (viewer-centered): "the man is to the right of the car"

🌍 Allocentric (object-centered): "the man is behind the car"

Such discrepancies can mislead robots in navigation, manipulation, or spatial reasoning tasks.

🛠️ Our Approach

We propose a Structured Spatial Representation that explicitly annotates key spatial elements:

🖼️ Scene descriptions

🏷️ Reference objects & orientations

🎯 Target objects & orientations

🔄 Reference frame types (egocentric / allocentric)

Based on this representation, we:

🗂️ Constructed a spatially annotated dataset

⚙️ Fine-tuned pre-trained VLMs using QLoRA

🔗 Integrated structured annotations into the model’s reasoning pipeline

📊 Results

Our method demonstrates:

⭐ Significant improvements over state-of-the-art models in spatial orientation reasoning tasks

🤖 Enhanced ability of VLMs to resolve egocentric–allocentric ambiguities

✅ More consistent and reliable outputs for robotics and multimodal AI

📄 Abstract

This study investigates the challenges of ambiguity faced by Visual-Language Models (VLMs) in understanding spatial semantics. Spatial cognition, influenced by cognitive psychology, spatial science, and cultural contexts, often assigns directionality to objects. For instance, while a car is inherently non-directional, human usage scenarios typically imbue it with an assumed orientation. In natural language, spatial relationship descriptions frequently omit explicit reference frame specifications, leading to semantic ambiguity. Existing VLMs, due to insufficient annotation of reference frames and object orientations in training data, often produce inconsistent responses. Consider an image where a car is positioned on the left side facing left and a man stands on the right side facing the viewer: an egocentric perspective describes the man as "to the right of the car," whereas an allocentric perspective interprets him as "behind the car," highlighting semantic discrepancies arising from different reference frames. Such ambiguities can lead to erroneous decisions when robots rely on natural language for navigation and manipulation. To address this problem, we propose a structured spatial representation method for identifying and annotating key spatial elements in images, including scene descriptions, reference objects and their orientations, target objects and their orientations, as well as reference frame types. Based on this representation, we constructed a dataset. By fine-tuning with QLoRA [1], these spatial elements were integrated into a pre-trained VLM. Experimental results demonstrate that our approach significantly outperforms state-of-the-art models in spatial orientation reasoning tasks, effectively enhancing the ability of VLMs to resolve spatial semantic ambiguities.

🏷️ Keywords

Visual-Language Models · Spatial Semantic Ambiguity · Reference Frame · Egocentric · Allocentric · Multimodal Reasoning

📄 Please view the full paper.
