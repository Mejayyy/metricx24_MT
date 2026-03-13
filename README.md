# MetricX-24 Improvements Study

This repository contains a proof-of-concept reproduction and analysis of MetricX-24, a regression-based Machine Translation (MT) evaluation framework originally developed by Google. Due to computational constraints, this project adapts the original architecture to smaller, more resource-efficient models (mT5-small and mT5-base) making it suitable for academic experimentation.


## 👥 Authors

This project was developed at the University of Bucharest, Faculty of Mathematics and Computer Science for the Artificial Intelligence course:

* Mihai-Radu Tătaru 


* Alexandru-Cristian Dogaru 


* Alexandru Jilavu 




---

## 🛠️ Tech Stack


**Python 3.10**: The primary programming language.



**PyTorch**: The underlying deep learning framework used for model construction and tensor operations.



**Hugging Face Transformers**: Used to load pre-trained mT5 models and manage the training loop via the Trainer API.



**Scikit-learn & SciPy**: Used for calculating evaluation metrics (like Pearson correlation) and data splitting.


 
**Pandas**: Utilized for efficient data preprocessing and handling JSONL dataset files.



---

## 📊 Dataset & Task


* **The Task**: We built a regression model acting as a "judge" to assign translation quality scores ranging from 0 (perfect) to 25 (bad).

  
* **The Data**: We utilized a subset of the WMT 2021 MQM (Multidimensional Quality Metrics) dataset.


* **Language Pairs**: English to German (en-de) and Chinese to English (zh-en).


* **Size**: Approximately 18,700 examples, which were split into a 9:1 ratio for training and validation.



---

## 🏗️ Architectures Explored

We explored different setups to determine the necessary complexity for effective quality estimation:

1. **Baseline (Random Chance)**: A baseline that guesses a random number between 0 and 25 to establish a minimum performance bar.


2. **MetricX-24 Replication (Encoder-Decoder)**: Modeled after the official MetricX-24 description, this uses the full mT5 model, modifying the decoder to output a single numerical score.


3. **Encoder-Only (Linear Head)**: Removes the decoder entirely to save resources, applying masked mean pooling to the encoder's embeddings and passing them through a linear layer.


4. **Encoder-Only (Sigmoid Head)**: Identical to the Linear Head, but adds a Sigmoid activation function to force predictions between 0 and 1, which are then multiplied by 25.



---

## ⚡ Training Optimizations

To ensure stable and efficient training on a single RTX 5090, we implemented several techniques:

* **Adafactor Optimizer**: A memory-efficient optimizer utilized in the original MetricX papers.


* **Inverse Square Root Schedule**: Dynamically adjusts the learning rate after a warm-up period.


* **Gradient Clipping**: Clips gradients to 1.0 to prevent individual confusing examples from heavily skewing the model's weights.


* **Early Stopping**: Halts training if no improvement is observed for 6 consecutive evaluations, preventing overfitting.


* **Mixed Precision (BF16/FP16)**: Lowers precision math to speed up training and save memory without sacrificing accuracy.



---

## 📈 Key Results

Our evaluations focused on Mean Squared Error (MSE), Pearson's r (correlation), and Pairwise Accuracy (ranking).

### Table 1: Effect of Architectural Choices on Trainable Parameters

| Pretrained Model | Architecture | Decoder | Parameters |
|---|---|---|---|
| mT5-small | Enc-Dec | yes | 300,176,768 |
| mT5-small | Enc-only + sigmoid | no | 146,941,121 |
| mT5-base | Enc-Dec | yes | 582,401,280 |
| mT5-base | Enc-only + sigmoid | no | 277,041,025 |



**Performance Summary:**

* Restricting training to an encoder-only setup reduces the model's trainable parameters by approximately 51%.


* The mT5-base family consistently achieves a lower MSE compared to the mT5-small variants.


* The **Encoder-Only with a Sigmoid head** represents the most favorable trade-off, combining strong ranking performance, near-optimal correlation, and substantially reduced computational cost.

### Table 2: Performance and Training Cost Across Architectural Variants

| Model | Architecture | MSE | Pearson | Pairwise Acc | Train time (s) |
|---|---|---|---|---|---|
| mT5-small | Enc-Dec | 12.12 | 0.739 | 0.681 | 847 |
| mT5-small | Enc-only | 12.22 | 0.735 | 0.683 | 591 |
| mT5-small | Enc-sigmoid | 13.34 | 0.724 | **0.732** | **414** |
| mT5-base | Enc-Dec | 11.98 | 0.745 | 0.682 | 2054 |
| mT5-base | Enc-only | 11.70 | **0.757** | 0.686 | 1515 |
| mT5-base | Enc-sigmoid | **11.60** | 0.753 | 0.697 | 751 |
| Random chance | - | 162.52


