# ResumeAtlas Resume Classification (PBL Project)

##  Project Overview
This project performs resume classification using both Classical Machine Learning and BERT-based deep learning models.

Dataset: ResumeAtlas (43 job categories, 13,389 resumes)

Models implemented:
- Bag of Words + Linear SVM
- TF-IDF + Linear SVM
- GloVe + Logistic Regression
- BERT (bert-base-uncased)



##  Final Results

| Model | Train Accuracy | Test Accuracy | Weighted F1 |
|------|---------------|--------------|-------------|
| BoW + SVM | 98.78% | 79.54% | 79.33% |
| TF-IDF + SVM | 98.32% | 82.19% | 81.68% |
| GloVe + Logistic Regression | 67.82% | 66.13% | 65.07% |
| BERT | 96.90% | 89.43% | 89.31% |


##  Dataset

Due to GitHub file size limits, dataset is hosted externally.

Download here:
https://drive.google.com/file/d/1zeFJTLe-CyP7rQ3XKu4WA-VIoTfg03Ij/view?usp=sharing



##  How to Run

1. Download dataset from link above
2. Place CSV in project folder
3. Run Jupyter notebook



##  Model Details

BERT Training:
- 7 epochs
- Cosine scheduler
- Warmup
- Gradient accumulation
- FP16 training
- Stratified split (70 / 10 / 20)



## Reference Paper
Resume classification using BERT on ResumeAtlas dataset.



##  Author
Yash Tyagi  
Manipal University Jaipur  
B.Tech CSE  
