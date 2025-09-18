# Healthcare Test Results Prediction Model

This project implements a deep learning model to predict **healthcare test results** using both **categorical embeddings** and **scaled numerical features**. The dataset contains diverse healthcare-related information such as patient demographics, medical conditions, hospital details, doctors, medications, billing, and admission records.

## Key Features
- **Data Preprocessing**:  
  - Removal of unnecessary columns.  
  - Label Encoding of categorical columns.  
  - Standard Scaling for numeric columns.  

- **Embeddings**:  
  Applied on categorical columns with large unique values (`Doctor`, `Hospital`) to capture hidden relationships.  

- **Model Architecture**:  
  - Embedding layers for high-cardinality features.  
  - Dense layers for deep learning.  
  - Concatenation of embeddings + numeric inputs.  
  - Output layer with `softmax` activation for classification.  

- **Training Strategy**:  
  - Optimizer: **Adam (lr=0.001)**  
  - Loss Function: **Sparse Categorical Crossentropy**  
  - Metric: **Accuracy**  
  - **EarlyStopping** callback with patience=5  

## Results
- **Train Accuracy**: 0.5871  
- **Test Accuracy**: 0.3691  

⚠️ Note: The gap between train and test accuracy shows that the model may be overfitting and requires further **hyperparameter tuning, regularization, or data balancing**.

## Requirements
The project uses:
- pandas  
- numpy  
- scikit-learn  
- tensorflow (keras)

All requirements are listed in `requirements.txt`.

---
🚀 This project demonstrates how **ANN with embeddings + numeric features** can be applied to real-world healthcare datasets for predictive modeling.
