# Multimodal Contrastive Learning for Image-Text Retrieval

![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-PyTorch%2C%20Transformers%2C%20Torchvision%2C%20FAISS-green)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📚 Overview

Welcome to the **Multimodal Contrastive Learning for Image-Text Retrieval** project! This repository implements a multimodal contrastive learning model inspired by [CLIP (Contrastive Language–Image Pretraining)](https://arxiv.org/abs/2103.00020) to align images and text in a shared embedding space. The goal is to enable efficient retrieval tasks where images can be matched with relevant captions and vice versa.

---

## 🎯 Objectives

- **Align Multimodal Data:** Learn a shared embedding space for images and text using contrastive learning techniques.
- **Efficient Retrieval:** Enable fast and accurate retrieval of images based on textual queries and captions based on image inputs.
- **Demonstrate Technical Proficiency:** Showcase skills in deep learning, computer vision, natural language processing, and system integration.

---

## 🚀 Key Achievements

- **Developed a Dual-Encoder Model:** Implemented separate encoders for images and text, projecting them into a shared embedding space.
- **Utilized Pre-trained Models Effectively:** Leveraged pre-trained DenseNet201 for image encoding and BERT-base-uncased for text encoding, enhancing performance and reducing training time.
- **Implemented Contrastive Learning Loss:** Employed a contrastive loss function to train the model to align image and text embeddings.
- **Achieved High Retrieval Accuracy:** Demonstrated effective image-to-text and text-to-image retrieval capabilities on the Flickr8k dataset.
- **Created Interactive Visualizations:** Developed functions to display images with top-matching captions and captions with top-matching images.
- **Ensured Scalability and Efficiency:** Optimized data loading and model inference for handling larger datasets and faster computations.
- **Maintained High Code Quality:** Wrote clean, modular code with comprehensive documentation for ease of understanding and future development.

---

## 🛠️ Technologies Used

- **Programming Language:** Python 3.7+
- **Deep Learning Frameworks:**
  - [PyTorch](https://pytorch.org/)
  - [Torchvision](https://pytorch.org/vision/stable/index.html)
  - [Hugging Face Transformers](https://huggingface.co/transformers/)
- **Data Processing and Visualization:**
  - [Pandas](https://pandas.pydata.org/)
  - [NumPy](https://numpy.org/)
  - [Matplotlib](https://matplotlib.org/)
- **Similarity Search:**
  - [FAISS](https://github.com/facebookresearch/faiss)
- **Dataset:**
  - [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)

---

## 🏗️ Model Architecture

The model consists of two primary components:

1. **Image Encoder**
2. **Text Encoder**

Both encoders are followed by projection layers to map the features into a shared embedding space of dimension `embed_dim` (set to 256).

### 🖼️ Image Encoder

- **Base Model:** Pre-trained **DenseNet201** from `torchvision.models`.
- **Modification:** Removed the classifier head to use it as a feature extractor.
- **Projection Layer:** Added a linear layer to project image features into the shared embedding space.

```python
self.image_encoder = models.densenet201(pretrained=True)
self.image_encoder.classifier = nn.Identity()
self.image_projection = nn.Linear(1920, embed_dim)
```

### 📝 Text Encoder
- **Base Model:** Pre-trained BERT-base-uncased from transformers.
- **Feature Extraction:** Used the hidden state corresponding to the [CLS] token.
- **Projection Layer:** Added a linear layer to project text features into the shared embedding space.

```python
self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)
```

### 🔗 Shared Embedding Space
- **Normalization:** Both image and text embeddings are L2-normalized to have a unit norm.
- **Contrastive Loss:** The model is trained using a contrastive loss function that brings matching image-text pairs closer and pushes non-matching pairs apart in the embedding space.

```python
image_embeddings = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)
text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)
```

### 📈 Results
#### Performance Metrics:
- **Training Loss:** Achieved a steady decrease in contrastive loss over 5 epochs.
- **Retrieval Accuracy:** Demonstrated high accuracy in retrieving relevant captions for images and relevant images for captions.
- **Embedding Quality:** The embeddings of matching image-text pairs have higher cosine similarity compared to non-matching pairs.

### 🔍 Applications
- **Image Search Engines:** Enhance search capabilities by retrieving images based on textual queries.
- **Automated Captioning:** Assist in generating captions for images by finding similar image-text pairs.
- **Content Recommendation Systems:** Recommend visual content based on user preferences expressed in text.
- **Multimodal Data Analysis:** Facilitate research in understanding and analyzing datasets containing both images and text.

