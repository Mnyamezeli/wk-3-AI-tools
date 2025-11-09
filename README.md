# Machine Learning Frameworks Project

This repository contains a collection of Jupyter notebooks demonstrating the implementation of various machine learning and deep learning tasks using different frameworks.

## Author
- **Name:** Mnyamezeli Voyi
- **GitHub:** [@Mnyamezeli](https://github.com/Mnyamezeli)

## Project Structure
```
.
├── Task_1_ML_Scikit_Learn.ipynb  # Machine Learning tasks using Scikit-learn
├── Task_2_DL_TensorFlow.ipynb    # Deep Learning tasks using TensorFlow
├── task_3_LNP.ipynb             # Language Processing tasks
├── Iris.csv/                    # Dataset directory
│   └── Iris.csv                # Iris dataset for machine learning tasks
└── tf_env/                     # TensorFlow virtual environment
```

## Tasks Overview

### 1. Machine Learning with Scikit-Learn (Task_1_ML_Scikit_Learn.ipynb)
- Implementation of machine learning algorithms using the Scikit-learn framework
- Focus on the Iris dataset classification
- Demonstrates data preprocessing, model training, and evaluation

### 2. Deep Learning with TensorFlow (Task_2_DL_TensorFlow.ipynb)
- Implementation of deep learning models using TensorFlow
- Neural network architecture and training
- Model evaluation and performance analysis

### 3. Language Processing (task_3_LNP.ipynb)
- Natural Language Processing tasks
- Text analysis and processing
- Language model implementation

## Environment Setup

The project uses a dedicated virtual environment (tf_env) with the following key dependencies:
- TensorFlow
- Scikit-learn
- Jupyter
- NumPy
- Pandas
- Various NLP libraries

## Dataset
The project includes the Iris dataset (`Iris.csv`), which is commonly used for machine learning classification tasks. The dataset contains:
- Multiple features of iris flowers
- Classification labels for different iris species

## Getting Started

1. Clone the repository
```bash
git clone https://github.com/Mnyamezeli/PLP-AI-for-Software-Engineering.git
```

2. Navigate to the project directory
```bash
cd PLP-AI-for-Software-Engineering/Specialazation_Assignments/wk_3_ML_Frameworks
```

3. Create and activate the virtual environment (Windows)
```bash
python -m venv tf_env
.\tf_env\Scripts\activate
```

4. Install required dependencies
```bash
pip install tensorflow scikit-learn jupyter pandas numpy
```

5. Launch Jupyter Notebook
```bash
jupyter notebook
```

## Usage
- Open each notebook in Jupyter to view and run the implementations
- Follow the instructions and comments within each notebook
- Execute cells sequentially to understand the workflow

## Visualizations and Results

### Task 1: Scikit-Learn Implementation
The Iris Species Classification project includes:
- Decision Tree Classifier implementation
- Model evaluation metrics:
  - Accuracy scores
  - Precision scores
  - Recall scores
  - Detailed classification report for each Iris species
- Visualizations of the dataset features and their relationships

### Task 2: TensorFlow Deep Learning
The MNIST Handwritten Digits Classification includes:
- CNN (Convolutional Neural Network) implementation
- Training visualizations:
  - Model accuracy plots (training vs validation)
  - Model loss plots (training vs validation)
- Sample predictions visualization showing:
  - Original handwritten digits
  - Model predictions
  - True labels
- Final test accuracy and loss metrics

### Task 3: Natural Language Processing
The NLP implementation includes:
- Named Entity Recognition (NER) for product reviews
- Sentiment Analysis results showing:
  - Entity extraction (Organizations and Products)
  - Sentiment classification (Positive/Negative/Neutral)
- Analysis of sample Amazon product reviews
- Entity and sentiment visualization for each review

## License
This project is part of the PLP AI for Software Engineering specialization.

## Acknowledgments
- Power Learn Project (PLP)
- AI for Software Engineering Program# wk-3-AI-tools
