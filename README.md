# Document Question Answering System

This project explores different approaches to document question answering (DocQA) — the task of answering natural language questions directly from complex documents such as PDFs, scanned forms, and academic research papers.

We implement and compare multiple pipelines:

- LayoutLM (fine-tuned transformer for document understanding)

- Retrieval-Augmented Generation (RAG) with Llama3 and Llava

- Qwen 2.5 VL (multimodal LLM fine-tuned for DocVQA)

Our goal is to evaluate these models’ strengths and limitations and determine whether hybrid approaches can outperform traditional fine-tuned models.

## Motivation

Academic and technical documents are often long and complex, with multi-column layouts, figures, tables, and mathematical notations. Manually extracting specific information is tedious and error-prone.

A DocQA system can streamline this process by enabling users to ask natural language questions and receive direct, accurate answers from documents.

## Project Structure
```python
Question-Answering/
│
├── layout_llm_scripts/      # LayoutLM fine-tuning & inference
├── llm_rag_pipeline/        # RAG pipeline using Llama3 + Llava
├── qwen_scripts/            # Qwen 2.5 VL scripts for DocVQA
├── README.md                # Documentation
```

## Datasets

We use a combination of benchmark datasets for training and evaluation:

- **SQuAD 2.0** → reading comprehension (Wikipedia-based QA, includes unanswerable questions)

- **FUNSD** → noisy scanned forms dataset (text + layout annotations)

- **DocVQA** → benchmark for document visual question answering on scanned documents

## Models & Pipelines
#### 1. LayoutLM Pipeline

- Base model: LayoutLM / LayoutLMv2

- Fine-tuned on FUNSD and DocVQA

- Incorporates textual + layout (and optionally visual) embeddings

- Training: batch size 8, 100 epochs

Workflow:

    1. Extract text from documents using OCR (e.g., Tesseract)

    2. Provide text, bounding boxes, and page image to LayoutLM

    3. Model predicts start/end tokens for answers

#### 2. RAG Pipeline

- **Retriever:** splits documents into text, tables, and image summaries

- **Vector Store:** embeddings stored for efficient retrieval

- **Generator:** Llama3 --> fine-tuned on SQuAD

- **Image summarization:** Llava 1.6

Pipeline:

    1. Break down PDFs into text, tables, and images

    2. Store embeddings in FAISS/vector DB

    3. Query → similarity search → context retrieved

    4. Pass to Llama3 for QA

#### 3. Qwen 2.5 VL Pipeline

- Multimodal LLM by Alibaba (7B params, trained on 4T multimodal tokens)

- Fine-tuned for DocVQA

- Handles text and image inputs jointly (no preprocessing required)

- Prompting uses ChatML format with special tokens (<|im_start|>, <|vision_start|>, etc.)

### Results

- **LayoutLM:** Performs well on structured form-like documents but struggles with long, text-heavy research papers.

- **RAG (Llama3 + Llava):** Complexity issues and limited image understanding. Difficult to fine-tune effectively.

- **Qwen 2.5 VL:** Out-of-the-box accuracy ~92% F1 on DocVQA without fine-tuning. Fast inference (0.4s per text-only page; 3x slower for image-heavy pages).

### Conclusion: 
**Qwen 2.5 VL** provides the best tradeoff between accuracy, efficiency, and ease of use.

### Installation

Clone the repository:
```python
git clone https://github.com/Aparna-J-Nair/Question-Answering.git
cd Question-Answering
```

### Challenges

- Fine-tuned LayoutLM struggles with long, text-heavy documents.

- Llava image summarization proved insufficient for robust RAG.

- Prompting strategies are crucial for maximizing Qwen’s output quality.
