# 🌟 From-Classical-NLP-Models-to-Large-Language-Models-for-Sentiment-Analysis

# 📄 Project Description and Structure

## 📝 Project Description

This project was conducted as a **freelance mission for a client in the United States** between **October and December**, lasting **3 months**.

**🎯 Objective:**
The main objective of the project is to **develop an efficient sentiment analysis system** capable of detecting emotions in text data using both classical deep learning models and large language models (LLMs).

**❗ Problem Statement:**
The client needed a solution to automatically classify customer feedback, social media posts, and other textual data into emotions such as sadness, joy, anger, fear, love, and surprise. Traditional rule-based or small-scale models were insufficient to capture the complex language patterns and nuances.

**💡 Proposed Solution:**

* Implement **supervised deep learning models** (RNN, LSTM, BERT) for high-accuracy sentiment classification.
* Apply **LLM approaches** using zero-shot, few-shot, and fine-tuning techniques to enhance flexibility and reduce the need for extensive labeled data.
* Provide a modular and reusable architecture for both experimentation and deployment.

---

## 📁 Project Structure

The project is organized in a **modular architecture** to separate data processing logic, model definitions, and execution scripts.

```
Analyse/                  # Data exploration tools
  Analyse.ipynb            # Notebook for exploratory data analysis
  Function.py              # Utility functions for data cleaning and processing

Architecture/             # Deep Learning model definitions
  RNN.py                   # Recurrent Neural Network implementation
  LSTM.py                  # Long Short-Term Memory model
  Transformers.py          # BERT-based classifier implementation

Data/                     # Raw and processed data files
  train.txt, val.txt, test.txt  # Raw datasets
  output.json              # Processed data or results in JSON format

DataSet/                  # PyTorch dataset management
  SentimentDataset.py      # Dataset class for sentiment handling
  TestDataset.py           # Dataset class for evaluation

LLM/                      # Advanced experiments with Large Language Models
  Fine-tuning.ipynb        # Scripts for model fine-tuning
  Zero_Shot_and_Few_Shot.ipynb  # Zero-shot and few-shot experiments

lora-mistral/             # LoRA adaptation folder for Mistral model

Main/                     # Main scripts for training and testing
  main_RNN.py
  main_LSTM.py
  main_Bert.py
  train.py
  train_Bert.py
  test.py
  test_Bert.py

Models/                   # Trained model binaries
  rnn_model.pth
  lstm_model.pth
  Bert_model.pth

config.py                 # Central configuration for hyperparameters (learning rate, batch size, epochs, etc.)
```

This structure ensures **clear separation of concerns**, easy **reproducibility**, and facilitates **future extensions** for both deep learning and LLM-based sentiment analysis.

---

## ⚙️ Installation

### 1. Create a Python environment (recommended)

```bash
conda create -n nlp-llm python=3.10 -y
conda activate nlp-llm
```

### 2. Install PyTorch and TorchText

```bash
# CPU-only
pip install torch==2.0.1+cpu torchvision torchaudio torchtext -f https://download.pytorch.org/whl/torch_stable.html

# GPU (CUDA 11.8)
pip install torch==2.0.1+cu118 torchvision torchaudio torchtext -f https://download.pytorch.org/whl/torch_stable.html
```

### 3. Install Hugging Face Transformers and Datasets

```bash
pip install transformers datasets
```

### 4. Install PEFT and accelerate for LoRA fine-tuning

```bash
pip install peft accelerate
```

### 5. Install other useful dependencies

```bash
pip install scikit-learn pandas numpy tqdm
```

### 6. Optional: Hugging Face Hub for login and models

```bash
pip install huggingface_hub
```

---

# 🧠 Part I – Supervised Sentiment Classification (RNN, LSTM, BERT)

## Introduction

The goal of this first part of the project is to implement different Natural Language Processing (NLP) models for **sentiment detection in sentences**. We implemented three types of models: **RNN, LSTM, and Transformer (BERT)**, and compared their performance.

The targeted sentiments are:

* `sadness` : 0
* `joy` : 1
* `anger` : 2
* `fear` : 3
* `love` : 4
* `surprise` : 5

The obtained performances are as follows:

* **RNN**: 34.95% accuracy
* **LSTM**: 77% accuracy
* **BERT**: 92.60% accuracy

## Features

### 🛠️ Data Preprocessing

* **Stop words removal** using `remove_stopwords` (library: `parsing.preprocessing`)
* **Sentence adjustment**: add `<PAD>` to shorter sentences and truncate longer sentences
* **Vocabulary construction**: build a unique vocabulary for all datasets (train, validation, test)
* **Sentence transformation**: convert sentences into indices based on the vocabulary
* **One-hot encoding** of words during DataLoader iteration

### 📦 DataLoader

Each batch contains:

* the text transformed into vectors
* the corresponding labels for each sentence

## Implemented Models

### 1️⃣ RNN (Recurrent Neural Network)

**Architecture**:

* `I2e`: input → embedding
* `Concat2h`: combines embedding and previous hidden state
* `H2o`: hidden → output
* `LogSoftmax`: final activation function

**Functioning**: Forward pass transforms the input into embeddings, combines it with the hidden state, applies ReLU, and outputs through softmax.

### 2️⃣ LSTM (Long Short-Term Memory)

**Architecture**:

* `Embedding_layer`: embedding vector of size `emb_size`
* `Forget_layer`: decides which information to forget
* `Input_layer`: computes new information to add to the cell
* `Output_layer`: computes the final output
* `H2end`: final linear layer + softmax

**Functioning**: Captures long-term dependencies and regulates the information flow in the hidden cell.

### 3️⃣ Transformer (BERT)

**Transfer Learning Approach**:

* **Backbone**: pre-trained BERT model
* **Head**: task-specific classification layer

**Data Flow**:

* `TextDataset` + `DataLoader` to manage memory and batch processing

**Training Loop**:

* Forward → Loss → Backward → Optimization

**Validation**: strict train/val/test split + early stopping to prevent overfitting

## Training Steps

1. Load and transform the data
2. Define the model (RNN, LSTM, or BERT)
3. Define the loss function and optimizer

**Phases**:

* **Training**: adjust model weights
* **Validation**: control overfitting
* **Testing**: final performance evaluation

## 📊 Results

| Model | Accuracy |
| ----- | -------- |
| RNN   | 34.95 %  |
| LSTM  | 77 %     |
| BERT  | 92.60 %  |

# 🤖 Part II – Sentiment Classification Using Large Language Models (Zero-Shot, Few-Shot, Fine-Tuning)

## Introduction

This part of the project focuses on **using large language models (LLMs) for sentiment classification** in sentences. Unlike traditional supervised models, LLMs allow **zero-shot, few-shot, and fine-tuning approaches** to predict the sentiment without training from scratch.

We use the **Mistral-7B-Instruct-v0.2** model from Hugging Face for text generation and sentiment classification.

The targeted sentiments remain the same:

* `sadness` : 0
* `joy` : 1
* `anger` : 2
* `fear` : 3
* `love` : 4
* `surprise` : 5

---

## Features

### ⚡ Zero-Shot and Few-Shot Classification

* Zero-shot: classify the emotion of a sentence using an instruction prompt without examples.
* Few-shot: include several examples of sentences with their emotions in the prompt to guide the model.

### 🛠️ Fine-Tuning

* Dataset preparation: convert training data into JSON with fields `instruction`, `input` (sentence), and `output` (label).
* Tokenization: tokenize input, instruction, and output together for causal language modeling.
* LoRA configuration: apply Low-Rank Adaptation (LoRA) for efficient fine-tuning.
* Training: use the Hugging Face `Trainer` to fine-tune the model with specified hyperparameters (batch size, learning rate, epochs, etc.).
* Model saving: save the fine-tuned model for later inference.

### 🔍 Inference

* Load the fine-tuned model and tokenizer.
* Use a pipeline for text generation to predict the emotion of new sentences.
* Extract the predicted emotion from the generated text.

---

## Training and Inference Steps

1. Prepare the training dataset in JSON format.
2. Load the pre-trained model and tokenizer.
3. Tokenize the dataset and apply LoRA for fine-tuning.
4. Train the model using the `Trainer` API.
5. Save the fine-tuned model.
6. Load the model for inference.
7. Construct prompts and generate the emotion predictions for new sentences.

---

## Notes

* Fine-tuning with LoRA allows adapting the large language model efficiently with limited resources.
* Zero-shot and few-shot methods allow using the model without extensive labeled data.
* The instruction prompt should clearly define the classification task and possible sentiment labels.

---

## 📝 Example Predicted Sentiments

| Input Sentence                                              | Predicted Sentiment |
| ----------------------------------------------------------- | ------------------- |
| I woke up feeling a little depressed                        | sadness             |
| I feel is he generous                                       | joy                 |
| I am feeling better though I don’t sound it                 | joy                 |
| I feel irritated and rejected without anyone doing anything | anger               |
| I was pregnant and felt terrified about having another baby | fear                |




#   F r o m - C l a s s i c a l - N L P - M o d e l s - t o - L a r g e - L a n g u a g e - M o d e l s - f o r - S e n t i m e n t - A n a l y s i s  
 #   F r o m - C l a s s i c a l - N L P - M o d e l s - t o - L a r g e - L a n g u a g e - M o d e l s - f o r - S e n t i m e n t - A n a l y s i s  
 #   F r o m - C l a s s i c a l - N L P - M o d e l s - t o - L a r g e - L a n g u a g e - M o d e l s - f o r - S e n t i m e n t - A n a l y s i s  
 