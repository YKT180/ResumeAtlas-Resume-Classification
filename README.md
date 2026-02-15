# ResumeAtlas Resume Classification

This project implements automated resume classification using both classical machine learning and transformer-based deep learning models.

The objective is to classify resumes into job categories based on textual content.



##  Dataset

**Dataset:** ResumeAtlas  
**Total Samples:** 13,389 resumes  
**Number of Classes:** 43 job categories  

Each sample contains:
- Resume text
- Category label



##  Data Split (Stratified)

| Split | Samples |
|---|---|
| Train | 9,372 |
| Validation | 1,339 |
| Test | 2,678 |

Split ratio follows research paper methodology:

- 70% Train
- 10% Validation
- 20% Test



##  Models Implemented

### 1️⃣ Classical Machine Learning

####  Bag of Words + Linear SVM
- CountVectorizer (20k features)
- Linear SVM classifier

####  TF-IDF + Linear SVM
- TF-IDF vectorization
- Hyperparameter tuning (C optimization)

####  GloVe Embeddings + Logistic Regression
- 100-dimensional GloVe embeddings
- Averaged word embeddings per resume



### 2️⃣ Deep Learning

####  BERT (bert-base-uncased)
- HuggingFace Transformers
- Fine-tuned for sequence classification
- FP16 training
- Cosine learning rate scheduler



##  Evaluation Metrics

For all models:

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- Macro F1-score
- Confusion matrix



##  Final Results

| Model | Train Accuracy | Test Accuracy | Test Weighted F1 |

| BoW + Linear SVM | 98.79% | 79.50% | 79.34% |
| TF-IDF + Linear SVM | 98.36% | 82.11% | 81.62% |
| TF-IDF + Linear SVM (Tuned) | 97.47% | **82.26%** | 81.70% |
| GloVe + Logistic Regression | 67.81% | 66.13% | 65.07% |
| BERT (bert-base-uncased) | 96.90% | **89.43%** | **89.31%** |



##  Macro F1 Scores

| Model | Macro F1 |

| BoW + Linear SVM | 78.28% |
| TF-IDF + Linear SVM | **81.19%** |
| GloVe + Logistic Regression | 63.72% |



##  Observations

- TF-IDF performs better than Bag-of-Words
- GloVe underperforms due to information loss from averaging embeddings
- BERT significantly outperforms classical models
- Classical models show overfitting (very high train accuracy vs test)



##  Environment

- Python
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- transformers
- PyTorch

GPU training used for BERT (RTX 4060).



##  Research Objective

Reproduce and evaluate classical and deep learning approaches for resume classification and compare performance under identical dataset and split conditions.


##  Author

Yash Tyagi  
Manipal University Jaipur  
CSE Undergraduate



##  Status

Project completed and ready for academic presentation and research reporting.


