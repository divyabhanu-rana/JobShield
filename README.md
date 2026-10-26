# JobShield
## Fake Job Posting Detection Using Deep Neural Networks

**FINAL PROJECT SUBMISSION**

**Course:** Artificial Neural Networks and Deep Learning  
**Course Code:** BDA404-5N

**Submitted By:** Divyabhanu Rana  
**Register Number:** 24215209  
**Class:** 5BDA

**Course Instructor:** Dr. Ramesh Chandra Poonia

---

## Project Overview

JobShield is my final project developed as part of the course **Artificial Neural Networks and Deep Learning (BDA404-5N)**.

The project focuses on detecting potentially fraudulent online job advertisements using a Deep Neural Network (DNN). It applies concepts of artificial neural networks, supervised learning, natural language processing and binary classification to address a real-world problem in online recruitment.

The system analyses both textual and structured information from job advertisements and produces an estimated fraud score. A trained neural network classifies each advertisement as either potentially fraudulent or not flagged as fraudulent.

The project covers the complete deep-learning workflow, including exploratory data analysis, data preprocessing, feature engineering, neural network implementation, model training, performance evaluation and application development.

A web-based application has been developed using Streamlit to demonstrate real-time predictions using the trained model.

**Important:** JobShield is designed as an academic fraud-screening system. Its predictions indicate patterns associated with fraudulent advertisements and do not conclusively establish whether a particular job advertisement or employer is genuine.

## 1. Problem Statement

The increasing use of online recruitment platforms has made job opportunities more accessible. However, these platforms may also contain fraudulent advertisements intended to mislead job seekers.

Such advertisements may contain misleading employment offers, unrealistic salary promises, incomplete company information or requests for payment.

Manually verifying large numbers of advertisements is difficult and time-consuming.

The primary problem addressed in this project is:

**Can a Deep Neural Network learn patterns from historical job advertisements and identify potentially fraudulent postings using textual and structured features?**

JobShield investigates this problem through supervised deep learning and evaluates the resulting model using appropriate classification metrics.

## 2. Project Objectives

The main objectives of this project are:

1. To study and analyse the Employment Scam Aegean Dataset (EMSCAD).
2. To identify missing information, duplicate records and class imbalance.
3. To implement a reproducible data preprocessing pipeline.
4. To transform textual and categorical information into numerical features.
5. To implement a Deep Neural Network using TensorFlow/Keras.
6. To compare the DNN against a Logistic Regression baseline.
7. To evaluate classification performance using accuracy, precision, recall, F1-score, ROC-AUC and Average Precision.
8. To investigate the effect of classification-threshold selection.
9. To develop a working web-based application using Streamlit.
10. To demonstrate the application through practical test cases and analyse its limitations.

## 3. Dataset Description

### 3.1 Dataset Source

The project uses the **Employment Scam Aegean Dataset (EMSCAD)**, a publicly available dataset created for research into fraudulent online recruitment advertisements.

The original dataset contains:

| Property | Value |
|---|---:|
| Total job advertisements | 17,880 |
| Legitimate advertisements | 17,014 |
| Fraudulent advertisements | 866 |
| Total attributes | 18 |
| Fraudulent proportion | 4.84% |

### 3.2 Target Variable

The dataset contains a binary target variable named `fraudulent`.

- `0` represents a legitimate advertisement.
- `1` represents a fraudulent advertisement.

The objective is to predict this target using information available in the advertisement.

### 3.3 Types of Features

The dataset contains three main categories of features.

**Textual features:**
- Job title
- Company profile
- Job description
- Requirements
- Benefits

**Categorical features:**
- Location
- Department
- Salary range
- Employment type
- Required experience
- Required education
- Industry
- Job function

**Binary features:**
- Telecommuting
- Company logo availability
- Screening questions availability

The dataset also contains a unique job identifier, which was excluded from model training.

## 4. Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the structure, distribution and characteristics of the dataset before model development.

The analysis included:

- Dataset dimensions and data types
- Missing-value analysis
- Duplicate-record identification
- Target-class distribution
- Numerical and categorical feature distributions
- Text-length analysis
- Comparison between legitimate and fraudulent advertisements

### 4.1 Class Imbalance

The dataset is highly imbalanced, with legitimate advertisements accounting for approximately 95.16% of the original records.

This imbalance is significant because a model that predicts every advertisement as legitimate could achieve high accuracy without identifying fraudulent advertisements.

Therefore, accuracy alone was not considered sufficient for evaluating model performance.

Additional emphasis was placed on precision, recall, F1-score and Average Precision.

### 4.2 Missing Information

EDA identified substantial missing information in several fields, particularly salary range, department and company profile.

Rather than removing all incomplete records, appropriate missing-value handling was implemented to preserve the available information.

### 4.3 Duplicate Records

A total of 281 duplicate advertisements were identified when the unique job identifier was excluded.

These repeated records were removed before splitting the dataset to reduce the risk of identical advertisements appearing in different data subsets.

## 5. Data Preprocessing

Data preprocessing was performed to prepare the dataset for numerical feature extraction and neural network training.

### 5.1 Missing-Value Handling

The following strategies were applied:

- Missing textual information was replaced with empty strings.
- Missing categorical information was replaced with the value `"Missing"`.
- Binary attributes were validated to ensure appropriate numerical values.
- Target labels were validated before splitting.

Incomplete advertisements were retained because missing information may itself be associated with patterns present in fraudulent postings.

### 5.2 Duplicate Removal

After removing repeated advertisements, the prepared dataset contained:

| Property | Records |
|---|---:|
| Original records | 17,880 |
| Duplicates removed | 281 |
| Final records | 17,599 |
| Legitimate records | 16,743 |
| Fraudulent records | 856 |

### 5.3 Feature Exclusion

The `job_id` attribute was excluded because it functions as a record identifier rather than a meaningful predictor of recruitment fraud.

### 5.4 Dataset Splitting

The processed dataset was divided using stratified sampling with `random_state=42`.

| Dataset | Number of Records | Proportion |
|---|---:|---:|
| Training | 12,319 | 70% |
| Validation | 2,640 | 15% |
| Testing | 2,640 | 15% |
| **Total** | **17,599** | **100%** |

Stratification was used to preserve approximately the same target-class distribution across the three subsets.

### 5.5 Prevention of Data Leakage

Special attention was given to preventing data leakage.

Feature extraction components that learn from the dataset were fitted using training data only.

The same fitted transformations were subsequently applied to validation data, test data and new advertisements submitted through the application.

The final test dataset was reserved for evaluating the selected model after development and threshold selection.

## 6. Feature Engineering

The project combines textual and structured features into a single numerical representation.

### 6.1 TF-IDF Vectorisation

Term Frequency-Inverse Document Frequency (TF-IDF) was used to convert textual information into numerical features.

TF-IDF assigns weights to words based on their importance within individual documents relative to the overall collection.

The following textual attributes were processed:

1. Job title
2. Company profile
3. Job description
4. Requirements
5. Benefits

Unigrams and bigrams were included to represent individual words and selected two-word combinations.

### 6.2 Categorical Encoding

Categorical attributes were converted into numerical representations using One-Hot Encoding.

The encoder was configured to handle previously unseen categories during inference.

### 6.3 Binary Features

The three binary attributes were included directly in the combined feature representation.

### 6.4 Final Feature Representation

After fitting the preprocessing pipeline, the final numerical representation contained **3,868 input features**.

This feature representation was used consistently for both Logistic Regression and the Deep Neural Network.

The fitted preprocessing transformer was saved as:

`models/preprocessor.joblib`

## 7. Baseline Model — Logistic Regression

Before implementing the Deep Neural Network, I developed a Logistic Regression model as a baseline.

The purpose of the baseline was to establish a reference against which the DNN could be evaluated.

Logistic Regression was selected because it is suitable for binary classification and can efficiently process sparse text-based numerical features.

Class balancing was enabled to account for the unequal distribution of legitimate and fraudulent advertisements.

The baseline model was trained using the same feature representation subsequently used for the DNN.

Its performance was evaluated using classification metrics and a confusion matrix.

The trained baseline model was saved as:

`models/logistic_baseline.joblib`

## 8. Deep Neural Network Implementation

The principal component of JobShield is a feed-forward Deep Neural Network developed using TensorFlow/Keras.

### 8.1 Network Architecture

The implemented architecture consists of:

| Layer | Configuration |
|---|---|
| Input | 3,868 features |
| Hidden Layer 1 | Dense, 128 neurons, ReLU |
| Regularisation | Dropout, 30% |
| Hidden Layer 2 | Dense, 64 neurons, ReLU |
| Regularisation | Dropout, 20% |
| Output | Dense, 1 neuron, Sigmoid |

**Total trainable parameters:** 503,553

### 8.2 Activation Functions

**ReLU (Rectified Linear Unit)** was used in the hidden layers to introduce non-linearity and support efficient learning.

**Sigmoid** was used in the output layer to generate a continuous score between 0 and 1 for binary classification.

### 8.3 Loss Function

Binary cross-entropy was used as the loss function because the project involves predicting one of two target classes.

### 8.4 Optimiser

The Adam optimiser was selected for training the DNN.

Adam adjusts parameter updates using estimates of the first and second moments of the gradients.

### 8.5 Regularisation

Dropout and L2 regularisation were incorporated to reduce the risk of overfitting.

### 8.6 Class Imbalance Handling

Class weights were calculated using the distribution of target labels in the training dataset.

A higher weight was assigned to the minority fraudulent class so that misclassification of fraudulent advertisements would have greater influence during training.

## 9. Model Training

The DNN was trained using the prepared training dataset and monitored using validation data.

### Training Configuration

| Parameter | Configuration |
|---|---|
| Framework | TensorFlow/Keras |
| Optimiser | Adam |
| Learning rate | 0.001 |
| Loss function | Binary cross-entropy |
| Batch size | 128 |
| Maximum epochs | 20 |
| Actual epochs executed | 8 |
| Class weighting | Enabled |
| Early stopping | Enabled |
| Learning-rate reduction | Enabled |

Early stopping monitored validation Average Precision through the Keras PR-AUC metric and restored the best model weights.

Training and validation accuracy and loss were recorded throughout training.

These results were used to generate actual learning curves for analysing model behaviour.

The trained DNN was saved as:

`models/jobshield_dnn.keras`

## 10. Classification Threshold Selection

The initial binary classification threshold was 0.50.

However, fraudulent job detection requires consideration of the trade-off between precision and recall.

A lower threshold may detect more fraudulent advertisements but may also incorrectly flag more legitimate advertisements.

Threshold analysis was therefore performed using validation predictions.

The selection procedure sought a recall of at least 95% while maximising F1-score among qualifying thresholds.

The selected threshold was:

**0.485**

On the validation dataset, this threshold achieved:

- Recall: 96.09%
- F1-score: 80.66%

The selected threshold was saved in:

`models/decision_threshold.json`

Threshold selection was completed before the final test evaluation.

## 11. Experimental Results and Model Evaluation

Both models were evaluated on the same held-out test dataset containing 2,640 advertisements.

The DNN used the validation-selected threshold of 0.485 while Logistic Regression used the default threshold of 0.50.

### 11.1 Final Test Results

| Performance Metric | Logistic Regression | Deep Neural Network |
|---|---:|---:|
| Accuracy | 97.50% | **97.92%** |
| Precision | 67.61% | **73.55%** |
| Recall | **92.97%** | 89.06% |
| F1-score | 78.29% | **80.57%** |
| ROC-AUC | **0.9932** | 0.9925 |
| Average Precision | 0.9181 | **0.9228** |

### 11.2 Interpretation

The DNN achieved higher accuracy, precision, F1-score and Average Precision than the Logistic Regression baseline.

However, Logistic Regression achieved higher recall and a slightly higher ROC-AUC.

This indicates that the DNN offered a better balance between precision and recall according to the F1-score, but the baseline detected a greater proportion of fraudulent advertisements at its evaluated threshold.

The findings demonstrate why comparing multiple performance metrics is necessary for imbalanced classification problems.

### 11.3 DNN Confusion Matrix

The final DNN confusion matrix on the test dataset contained:

| Classification Outcome | Count |
|---|---:|
| True Negatives | 2,471 |
| False Positives | 41 |
| False Negatives | 14 |
| True Positives | 114 |

The model correctly identified 114 of the 128 fraudulent advertisements.

However, 14 fraudulent advertisements were incorrectly classified as not flagged.

These false negatives are particularly important because potentially fraudulent opportunities may pass through the screening process.

The model also incorrectly flagged 41 legitimate advertisements.

### 11.4 Generalisation Performance

The validation recall at the selected threshold was 96.09%, while the final test recall was 89.06%.

This difference indicates that performance on unseen data may vary from validation performance.

Consequently, the reported metrics should be interpreted as estimates based on the specific held-out dataset rather than guarantees of performance on all future job advertisements.

## 12. JobShield Web Application

A web-based application was developed using Streamlit to demonstrate the trained DNN in a practical environment.

### 12.1 Application Features

The application allows users to:

- Enter a job title.
- Paste a job description.
- Provide company information.
- Enter job requirements and benefits.
- Supply additional structured job attributes.
- Submit the advertisement for analysis.
- View the estimated fraud score.
- View the resulting classification.
- View an illustrative screening level.

### 12.2 Prediction Workflow

The application follows this process:

**User Input → Data Cleaning → Saved Feature Transformer → Trained DNN → Sigmoid Score → Saved Decision Threshold → Classification**

The application uses the trained model and fitted preprocessing artefacts rather than fitting new transformations during inference.

### 12.3 Prediction Output

The system classifies submitted advertisements as:

- **Potentially Fraudulent:** The model score meets or exceeds the selected threshold.
- **Not Flagged:** The model score is below the selected threshold.

The application also displays LOW, MEDIUM or HIGH screening levels based on illustrative score bands.

These screening levels are not independently calibrated categories.

**Important:** The estimated fraud score is an uncalibrated neural-network output. It must not be interpreted as a verified probability that an advertisement is fraudulent.

## 13. Application Testing

The working application was tested using manually constructed job advertisements representing different input conditions.

### 13.1 Manual Test Results

| Test Case | Observed Model Score | Application Behaviour |
|---|---:|---|
| Normal-looking software developer advertisement | 0.74% | Not flagged |
| Suspicious-looking recruitment advertisement | 99.38% | Flagged |
| Incomplete office assistant advertisement | 99.69% | Flagged |
| Empty submission | Not applicable | Validation warning displayed |

All four tests demonstrated successful application behaviour without a reported crash.

These manually constructed examples were used to test application functionality rather than establish additional classification accuracy.

### 13.2 Observation

The incomplete office assistant advertisement received a high estimated fraud score despite containing limited information.

This observation suggests that incomplete or minimal advertisements may be strongly flagged by the model.

However, incomplete information does not necessarily indicate genuine fraud.

This behaviour highlights the importance of interpreting automated predictions carefully and using independent verification.

### 13.3 Testing Outcome

The manual tests confirmed that:

1. The application accepts normal textual inputs.
2. The application generates predictions using the saved DNN.
3. The stored classification threshold is applied.
4. Partial job information can be processed.
5. Empty submissions produce validation feedback instead of a prediction.
6. The interface displays the model output and relevant limitations.

## 14. Project Structure

The project is organised into separate components for data analysis, preprocessing, feature engineering, model development, evaluation and application deployment.

**Main project directory:** `FINAL PROJECT/`

| Folder/File | Purpose |
|---|---|
| `data/fake_job_postings.csv` | Original EMSCAD dataset |
| `data/processed/` | Prepared datasets and feature matrices |
| `notebooks/01_EDA.ipynb` | Exploratory data analysis |
| `notebooks/02_PreProcessing.ipynb` | Data cleaning and splitting |
| `notebooks/03_FeatureEngg.ipynb` | Feature extraction and baseline model |
| `notebooks/04_ANNModel.ipynb` | Deep Neural Network implementation and training |
| `notebooks/05_Evaluations.ipynb` | Threshold selection and final evaluation |
| `src/preprocessing.py` | Reusable feature preparation functions |
| `src/predict.py` | Model loading and prediction functions |
| `models/` | Trained models and preprocessing artefacts |
| `reports/` | Model evaluation metrics and results |
| `reports/figures/` | Training curves and evaluation visualisations |
| `app.py` | Streamlit application |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

This structure separates experimental notebooks from reusable application logic.

## 15. Installation and Execution

### 15.1 Requirements

The project requires Python 3 and the libraries listed in `requirements.txt`.

The principal libraries include:

- NumPy
- Pandas
- SciPy
- scikit-learn
- TensorFlow/Keras
- Matplotlib
- Joblib
- Streamlit
- Jupyter Notebook

### 15.2 Installation

Open Command Prompt in the main `FINAL PROJECT` directory and install the required libraries using:

`py -m pip install -r requirements.txt`

### 15.3 Application Execution

Launch the Streamlit application using:

`py -m streamlit run app.py`

The application normally opens at:

http://localhost:8501

### 15.4 Using the Application

1. Open JobShield in the browser.
2. Enter a job title and description.
3. Add company information, requirements and benefits if available.
4. Optionally provide structured job attributes.
5. Click **Analyse Job Posting**.
6. Examine the model classification and estimated fraud score.
7. Independently verify any advertisement before making employment-related decisions.

The trained model and preprocessing artefacts must be present in the `models` directory.

## 16. Technologies Used

| Technology | Application |
|---|---|
| Python | Primary programming language |
| Pandas | Dataset manipulation and cleaning |
| NumPy | Numerical operations |
| scikit-learn | Feature engineering, baseline model and evaluation |
| TensorFlow/Keras | DNN implementation and training |
| SciPy | Sparse feature representation |
| Matplotlib | Data analysis and performance visualisation |
| Joblib | Saving preprocessing and baseline artefacts |
| Streamlit | Web application development |
| Jupyter Notebook | Experimentation and documentation |

## 17. Project Limitations

Although JobShield demonstrates promising classification performance, several limitations remain.

1. **Historical dataset:** EMSCAD contains historical job advertisements and may not fully represent more recent recruitment scams.

2. **Class imbalance:** The dataset contains substantially fewer fraudulent records than legitimate records.

3. **False negatives:** The final DNN failed to flag 14 fraudulent advertisements in the held-out test dataset.

4. **False positives:** Some legitimate advertisements were incorrectly classified as potentially fraudulent.

5. **Incomplete information:** Legitimate advertisements with limited information may receive high fraud scores.

6. **Uncalibrated scores:** The neural network's sigmoid output has not undergone probability calibration.

7. **Employer verification:** The system does not independently verify company registration, employer identity or vacancy authenticity.

8. **Unknown attributes:** Missing binary attributes are currently represented as zero, which does not distinguish unknown values from confirmed absence.

9. **Generalisation:** Performance on newer datasets, different recruitment platforms and changing fraud patterns has not been established.

For these reasons, JobShield is intended to assist preliminary screening rather than replace human judgement or employer verification.

## 18. Future Scope

The project can be extended in several directions.

### 18.1 Expanded Dataset

Collecting more recent recruitment advertisements could improve the model's ability to recognise emerging fraud patterns.

### 18.2 Improved Text Representation

Advanced natural language processing techniques and pretrained language models could be investigated for improved contextual understanding.

### 18.3 Probability Calibration

Calibration methods could be evaluated to make the model's scores more interpretable.

### 18.4 Explainable Artificial Intelligence

Explainability techniques could be incorporated to identify which input features most strongly influence predictions.

### 18.5 Employer Verification

Future versions could incorporate verified company information and additional recruitment-platform signals.

### 18.6 Improved Missing-Value Handling

The system could distinguish between unknown information and confirmed negative attributes.

### 18.7 Model Monitoring

Periodic evaluation and retraining could help address changes in recruitment practices and fraudulent advertisement patterns.

## 19. Learning Outcomes

Through the development of JobShield, I applied the concepts studied in **Artificial Neural Networks and Deep Learning (BDA404-5N)** to a practical binary classification problem.

The project provided hands-on experience in:

- Understanding artificial neural networks and deep-learning architectures
- Preparing real-world datasets for supervised learning
- Handling class imbalance and preventing data leakage
- Applying TF-IDF and categorical feature encoding
- Implementing Dense layers, ReLU and sigmoid activations
- Understanding binary cross-entropy and Adam optimisation
- Using dropout, regularisation and early stopping
- Monitoring training and validation performance
- Interpreting confusion matrices and classification metrics
- Comparing a neural network with a simpler baseline
- Selecting a decision threshold using validation data
- Saving trained models and preprocessing artefacts
- Integrating a trained DNN into a working application
- Evaluating practical limitations of AI-based predictions

The project also strengthened my understanding of how a machine-learning experiment is developed into a functional software application.

## 20. Conclusion

JobShield was developed as my final project for **Artificial Neural Networks and Deep Learning (BDA404-5N)** to demonstrate the practical application of deep learning in detecting potentially fraudulent online job advertisements.

The project successfully integrated dataset analysis, preprocessing, text feature extraction, structured data encoding, neural network training, model comparison and application development.

The implemented DNN achieved **97.92% accuracy, 73.55% precision, 89.06% recall and 80.57% F1-score** on the held-out test dataset.

Compared with the Logistic Regression baseline, the DNN achieved a higher F1-score and precision, although the baseline obtained higher recall.

A functional Streamlit application was also developed to demonstrate predictions using the saved model and preprocessing pipeline.

The experimental results indicate that deep learning can be useful for identifying patterns associated with potentially fraudulent job advertisements. However, false negatives, incomplete-information sensitivity and the absence of independent employer verification remain important limitations.

Overall, this project demonstrates the application of artificial neural networks to a real-world classification problem while highlighting the importance of reliable evaluation, responsible interpretation and practical implementation.

## 21. References

1. Vidros, S., Kolias, C., Kambourakis, G. and Akoglu, L. (2017). ‘Automatic Detection of Online Recruitment Frauds: Characteristics, Methods, and a Public Dataset’. *Future Internet*, 9(1), 6. https://doi.org/10.3390/fi9010006

2. TensorFlow. *TensorFlow and Keras Documentation*. https://www.tensorflow.org/

3. Scikit-learn Developers. *Scikit-learn Documentation*. https://scikit-learn.org/

4. Streamlit. *Streamlit Documentation*. https://docs.streamlit.io/

---

**Project:** JobShield — Fake Job Posting Detection Using Deep Neural Networks

**Submitted by:** Divyabhanu Rana  
**Register Number:** 24215209  
**Class:** 5BDA

**Course:** Artificial Neural Networks and Deep Learning  
**Course Code:** BDA404-5N

**Course Instructor:** Dr. Ramesh Chandra Poonia

**Submission Type:** Final Project