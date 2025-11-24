# 🩺 Medical RAG LLM

## A Retrieval-Augmented Generation System for Medical Knowledge

This project implements a high-performance **Retrieval-Augmented Generation (RAG)** system specialized in answering queries based on private or domain-specific medical knowledge. 

[Image of Retrieval-Augmented Generation system flow]


### ⭐️ Key Features

* **Domain Specialization:** Leverages a custom vector store to ground LLM responses in specific medical literature.
* **Vector Database:** Utilizes **Qdrant** for high-performance vector search and management.
* **LLM Integration:** Powered by the **Gemini API** for embeddings and generative medical query responses.
* **Multimodal Readiness:** Includes structure for text, image, and audio preprocessing pipelines.

---

## 🚀 Getting Started

This guide provides the complete setup required to deploy and run the RAG system locally.

### Prerequisites

Ensure the following tools are installed on your system:

* **Git**
* **Python 3.9+**
* **Docker** (Essential for running the Qdrant service)

### 1. Repository Setup

Clone the project and prepare the environment:

```bash
# Clone the repository
git clone [https://github.com/sudhakar-1104/Medical_RAG_LLM.git](https://github.com/sudhakar-1104/Medical_RAG_LLM.git)
cd Medical_RAG_LLM

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate
