# XLM-R for Robust Cross-Lingual and Multilingual Sentiment Analysis of Code-Mixed E-Commerce Product Reviews

Official implementation of **"XLM-R for Robust Cross-Lingual and Multilingual Sentiment Analysis of Code-Mixed E-Commerce Product Reviews"**

📄 **Paper**: INCOSST 2026 (Camera-Ready)

---

## 📋 Overview

The rapid expansion of global e-commerce platforms has led to a surge of user-generated reviews written in multiple languages and code-mixed forms (e.g., Bengali written in Latin script mixed with English). This linguistic heterogeneity makes it difficult for conventional monolingual models to generalize, leading to inconsistent sentiment classification.

- **Background**: Multilingual and cross-lingual e-commerce reviews are increasingly common, but conventional monolingual sentiment models struggle to generalize across languages and code-mixed text.
- **Objective**: To evaluate whether multilingual pre-trained transformer models better preserve sentiment polarity through translation and code-mixing compared to monolingual and general-purpose alternatives.
- **Datasets**: MARC (Multilingual Amazon Reviews Corpus, 18,000 reviews across 6 languages) and BanglishRev (Bengali–English code-mixed e-commerce reviews, 128,543 products).
- **Methods**: Fine-tuning and benchmarking of BERT-base, RoBERTa, mBERT, and XLM-R, plus few-shot prompting of GPT-3.5, following a KDD (Knowledge Discovery in Database) pipeline with machine-translation-based normalization to English.
- **Key Result**: XLM-R consistently achieves the best performance across both datasets, reaching **0.97 train accuracy** and **0.95 test accuracy/F1-score** on MARC, with McNemar's test showing **no statistically significant difference** from mBERT (p = 0.3817).

### Key Features

- **Cross-Lingual & Code-Mixed Benchmark**: Evaluates 5 model families (BERT-base, RoBERTa, mBERT, XLM-R, GPT-3.5) on both a multilingual dataset (MARC) and a code-mixed dataset (BanglishRev).
- **KDD-Based Pipeline**: Structured five-stage pipeline — data selection, preprocessing, transformation (translation), data mining (model training), and evaluation.
- **Machine-Translation Normalization**: Uses GoogleTranslator (DeepTranslator library) to translate all multilingual/code-mixed text into English as a pivot language before classification.
- **Statistical Significance Testing**: McNemar's test used to determine whether performance differences between top models (XLM-R vs. mBERT) are statistically meaningful.
- **Error Analysis**: Qualitative analysis of translation-induced misclassifications on both datasets.

---

## 🎯 Key Contributions

1. **Comprehensive Benchmark**: Systematic comparison of conventional transformers (BERT-base, RoBERTa), multilingual pre-trained models (mBERT, XLM-R), and a large language model (GPT-3.5) for sentiment classification of code-mixed e-commerce reviews.
2. **Cross-Dataset Validation**: Evaluation across two structurally different datasets — a balanced multilingual dataset (MARC, 6 languages) and a real-world code-mixed dataset (BanglishRev, Bengali/English/Banglish).
3. **Statistical Rigor**: McNemar's test on 293 differing predictions between XLM-R and mBERT (p = 0.3817), showing both models achieve statistically comparable performance despite XLM-R's slightly higher metrics.
4. **Multilingual Robustness Framing**: Demonstrates that multilingual pre-training — not model scale alone — is the key factor for preserving sentiment signal across translation and code-mixing, with fine-tuned transformers outperforming non-fine-tuned GPT-3.5 (0.73 MARC / 0.68 BanglishRev test accuracy).
5. **State-of-the-Art Comparison**: Outperforms previously reported F1-scores for mBERT and XLM-R on multilingual (0.93 / 0.97 vs. 0.86 / 0.91 in prior work) and code-mixed (0.96 / 0.96 vs. 0.82 / 0.88 in prior work) settings.

---

## Project Architecture / Research Workflow

This study adopts the **KDD (Knowledge Discovery in Database) framework**, consisting of five stages: **Data Selection → Data Pre-processing → Data Transformation → Data Mining → Evaluation**.

<img width="1551" height="1014" alt="figure1_flow_of_proposed_method" src="https://github.com/user-attachments/assets/ce8f393a-a59c-42d0-ae3c-e26e173b45de" />


1. **Data Selection**: MARC (multilingual) and BanglishRev (code-mixed) datasets.
2. **Data Pre-processing**: Text normalization, lowercasing, URL/emoji removal, tokenization, sentence splitting, and FastText-based language identification.
3. **Data Transformation**: Translation of all text to English (pivot language) using GoogleTranslator (DeepTranslator), followed by an 80:10:10 train/validation/test split.
4. **Data Mining**: Fine-tuning of BERT-base, RoBERTa, mBERT, and XLM-R; few-shot prompting of GPT-3.5.
5. **Evaluation**: Accuracy, precision, recall, F1-score, and execution time, plus McNemar's test for statistical significance between top models.

---

## 🚀 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers (Hugging Face) 4.x+
- deep-translator (GoogleTranslator module)
- fasttext (for language identification)
- CUDA 11.8+ (GPU training; experiments were run on NVIDIA T4 via Google Colab Pro, High-RAM)
- scikit-learn, pandas, numpy
- statsmodels (for McNemar's test)
- openai (for GPT-3.5 few-shot inference)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-name>
```

2. Create a conda environment:
```bash
conda create -n xlmr-sentiment python=3.8
conda activate xlmr-sentiment
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation

### 1) MARC (Multilingual Amazon Reviews Corpus)

A multilingual text classification dataset obtained from the Hugging Face repository. In this study, **18,000 reviews** were sampled, evenly distributed across **six languages** — Japanese (ja), German (de), English (en), French (fr), Spanish (es), and Chinese (zh) — each contributing 16.7% of the total samples. Each record contains review text, review title, star rating, anonymized reviewer/product IDs, and a coarse-grained product category.

- **Source**: (https://huggingface.co/datasets/mteb/amazon_reviews_multi)
- **Total samples used**: 18,000 reviews
- **Split**: 80% train / 10% validation / 10% test

### 2) BanglishRev Dataset

A large-scale e-commerce product review corpus for sentiment analysis in Bengali, English, Banglish (Bengali written in Latin script), and mixed Bengali–English text, obtained from the Hugging Face repository.

- **Source**: (https://huggingface.co/datasets/BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset)
- **Total samples**: Review data curated from 128,543 products (out of >1,000,000 candidate products)
- **Language distribution**: Banglish 37.77%, English 31.27%, Bangla 25.04%, Code-mixed 5.92%
- **Split**: 80% train / 10% validation / 10% test

### Download

Once downloaded, organize the data as follows:
```
ProjectName/
├── data/
│   ├── marc/
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   └── banglishrev/
│       ├── train/
│       ├── validation/
│       └── test/
```

---

## 🏋️ Training

### Pre-processing & Translation
All non-English and code-mixed reviews are translated into English (pivot language) using the `GoogleTranslator` module from the `deep-translator` library prior to model training.

### Fine-tuning
```bash
python train.py --model xlm-roberta-base --dataset marc --epochs 5 --batch_size 32 --lr 2e-5
python train.py --model bert-base-multilingual-cased --dataset marc --epochs 5 --batch_size 32 --lr 2e-5
python train.py --model bert-base-uncased --dataset marc --epochs 3 --batch_size 32 --lr 2e-5
python train.py --model roberta-base --dataset marc --epochs 3 --batch_size 32 --lr 2e-5
```

### GPT-3.5 (Few-Shot Prompting, No Fine-Tuning)
```bash
python evaluate_gpt.py --model gpt-3.5-turbo --dataset marc --shots 3 --temperature 0.0 --max_tokens 10
```

### Hyperparameter Settings

| Model | Batch Size | Epochs | Learning Rate |
|-------|-----------|--------|----------------|
| XLM-R, mBERT | 32 | 5 | 2 × 10⁻⁵ |
| RoBERTa, BERT-base | 32 | 3 | 2 × 10⁻⁵ |

| GPT-3.5 Parameter | Value |
|---|---|
| Number of shots | 3 |
| Temperature | 0.0 |
| Max tokens | 10 |

**Hardware**: All experiments were conducted on Google Colab Pro with an NVIDIA T4 GPU (High-RAM configuration).

---

## 📊 Results

### Model Performance on MARC and BanglishRev

| Algorithm | Split | MARC Accuracy | MARC Precision | MARC Recall | MARC F1 | BanglishRev Accuracy | BanglishRev Precision | BanglishRev Recall | BanglishRev F1 |
|---|---|---|---|---|---|---|---|---|---|
| BERT-base | Train | 0.80 | 0.95 | 0.80 | 0.86 | 0.89 | 0.88 | 0.87 | 0.87 |
| BERT-base | Test | 0.77 | 0.95 | 0.78 | 0.86 | 0.85 | 0.83 | 0.84 | 0.83 |
| BERT-base | Validation | 0.80 | 0.95 | 0.80 | 0.87 | 0.86 | 0.84 | 0.85 | 0.84 |
| RoBERTa | Train | 0.73 | 0.73 | 0.73 | 0.73 | 0.87 | 0.86 | 0.85 | 0.85 |
| RoBERTa | Test | 0.72 | 0.72 | 0.72 | 0.72 | 0.83 | 0.81 | 0.82 | 0.81 |
| RoBERTa | Validation | 0.68 | 0.66 | 0.67 | 0.66 | 0.84 | 0.82 | 0.83 | 0.82 |
| mBERT | Train | 0.93 | 0.93 | 0.93 | 0.93 | 0.97 | 0.93 | 0.96 | 0.94 |
| mBERT | Test | 0.87 | 0.88 | 0.88 | 0.87 | 0.94 | 0.88 | 0.93 | 0.89 |
| mBERT | Validation | 0.90 | 0.90 | 0.89 | 0.89 | 0.94 | 0.87 | 0.93 | 0.89 |
| **XLM-R** | **Train** | **0.97** | **0.97** | **0.97** | **0.97** | **0.97** | **0.94** | **0.96** | **0.95** |
| **XLM-R** | **Test** | **0.95** | **0.95** | **0.95** | **0.95** | **0.95** | **0.90** | **0.93** | **0.88** |
| **XLM-R** | **Validation** | **0.95** | **0.95** | **0.95** | **0.95** | **0.96** | **0.92** | **0.95** | **0.93** |
| GPT-3.5 | Train | 0.75 | 0.76 | 0.74 | 0.75 | 0.72 | 0.74 | 0.70 | 0.71 |
| GPT-3.5 | Test | 0.73 | 0.74 | 0.72 | 0.73 | 0.68 | 0.69 | 0.66 | 0.67 |
| GPT-3.5 | Validation | 0.71 | 0.72 | 0.70 | 0.71 | 0.69 | 0.70 | 0.67 | 0.68 |

**Execution time** (approx.): BERT-base ~565–590s, RoBERTa ~616–646s, mBERT ~610–671s, XLM-R ~811–1738s, GPT-3.5 ~781–820s.

### Statistical Significance (McNemar's Test)
XLM-R vs. mBERT prediction outputs showed **293 differing predictions**, with a **p-value of 0.3817 (> 0.05)** — indicating **no statistically significant difference** between the two models, despite XLM-R's slightly higher metrics.

### Error Analysis
XLM-R showed low misclassification rates: **328 errors (2.99%) on MARC** and **306 errors (2.49%) on BanglishRev**. Errors were largely attributed to translation inaccuracies that altered semantic meaning or sentiment expressions.

### Comparison with Previous Studies (F1-Score)

| Model | Dataset Type | This Study | Previous Studies |
|---|---|---|---|
| mBERT | Multilingual | 0.93 | 0.86 / 0.77 |
| XLM-R | Multilingual | 0.97 | 0.91 / 0.80 |
| GPT-3.5 | Multilingual | 0.74 | 0.87 |
| mBERT | Code-Mixed | 0.96 | 0.82 |
| XLM-R | Code-Mixed | 0.96 | 0.88 |
| GPT-3.5 | Code-Mixed | 0.70 | 0.86 |

Both mBERT and XLM-R outperformed previously reported results in multilingual and code-mixed settings, while GPT-3.5 (used without fine-tuning) underperformed relative to prior fine-tuned/few-shot baselines.

---

## 🏗️ Project Structure

```
ProjectName/
├── data/                       # MARC and BanglishRev datasets
├── data_loader/                 # Dataset loaders and preprocessing
│   ├── marc_loader.py
│   └── banglishrev_loader.py
├── preprocessing/                # Text normalization, translation, language ID
│   ├── clean_text.py
│   ├── translate_to_english.py   # GoogleTranslator (deep-translator) pipeline
│   └── language_id.py            # FastText-based language identification
├── model/                        # Fine-tuning configs per model
│   ├── xlmr_model.py
│   ├── mbert_model.py
│   ├── bert_model.py
│   └── roberta_model.py
├── llm/                           # GPT-3.5 few-shot prompting
│   └── gpt35_prompting.py
├── evaluation/                    # Evaluation utilities
│   ├── metrics.py                 # Accuracy, precision, recall, F1
│   ├── mcnemar_test.py            # McNemar's significance test
│   └── error_analysis.py
├── figures/                        # Paper figures (pipeline, confusion matrices, etc.)
├── train.py                        # Training script (transformer models)
├── evaluate_gpt.py                 # GPT-3.5 evaluation script
└── requirements.txt                 # Python dependencies
```

---

## 🙏 Acknowledgments

- Big Data Lab, Information Systems Study Program, Faculty of Engineering and Informatics, Universitas Multimedia Nusantara (UMN)
- Supervisor: Dr. Dinar Ajeng Kristiyanti, S.Kom., M.Kom.
- The authors thank Universitas Multimedia Nusantara for its financial and institutional support, which made this research possible.

---

## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: frans.imanuel1@student.umn.ac.id

---

## 📜 License

This project is released under the MIT License. See [LICENSE](LICENSE) file for details.
