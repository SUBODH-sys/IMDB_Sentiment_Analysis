# IMDB_Sentiment_Analysis
This project implements a text classification pipeline for the IMDb Movie Reviews Dataset to classify movie reviews as positive or negative, Two models are implemented:
1. Logistic Regression: A fast, interpretable baseline using TF-IDF vectorization.
2. BERT: A pretrained transformer model (bert-base-uncased) fine-tuned for higher accuracy, addressing the bonus requirement.
The pipeline includes data loading, preprocessing, model training, and evaluation, with metrics (Accuracy, Precision, Recall, F1-Score) and confusion matrices for both models.

Dataset Link :- 'https://drive.google.com/file/d/1iV4OxEeRObYbO9MshRmlzJqVgC54qdWn/view?usp=sharing'

## Usage
1. Clone or Download the notebook and dataset zip
2. Install Dependencies
   pip install numpy==1.26.4 pandas scikit-learn transformers datasets torch matplotlib seaborn
3. Run the Notebook, make sure to update file paths. Use Kaggle Editor for BERT training.
4. Run the Streamlit app with this command in terminal : "streamlit run d:/DeviXy/app.py"

*Demo Link* : "https://drive.google.com/file/d/1y53Mw_B0IqhHiS3g9ufazvkdd3bF0inV/view?usp=sharing"

## Data Distribution
![Screenshot 2025-07-01 145307](https://github.com/user-attachments/assets/40b66986-d315-437c-b68d-b6f1186a98ad)

![image](https://github.com/user-attachments/assets/30edb940-a0ba-425f-bedb-a00f0173c2ac)

## Model Metrics
| Model             | Accuracy | Precision | Recall | F1-Score |
|-------------------|----------|-----------|--------|----------|
| Logistic Regression| 0.88   | 0.87      | 0.88   | 0.87     |
| BERT              | 0.90      | 0.91         | 0.90      | 0.90     |

## Logistic Regression Confusion Matrix
![image](https://github.com/user-attachments/assets/85d4b2d6-650b-44bc-94e1-e14359f83320)

## BERT Confusion Matrix
![Screenshot 2025-07-01 145111](https://github.com/user-attachments/assets/0b6bcf53-2cc6-4a1e-88c3-1dc86b62a04f)


