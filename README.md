# Ecommerce Product Description Classifier

## Overview
This project aims to develop a text classification model to categorize products into different classes based on their descriptions. The goal is to build a machine learning model that can accurately classify products into predefined categories such as Electronics, Fashion, Home, and Sports.

## Project Structure
The project is organized as follows:

```
project_root/
│
├── data/
│   ├── processed/  # Processed data files
│   └── generated/  # Generated dataset
│
├── models/
│   ├── saved_models/  # Saved model files
│   └── model_evaluation/  # Evaluation metrics, model performance
│
├── notebooks/  # Jupyter notebooks for experimentation and documentation
│
├── src/
│   ├── data_processing.py  # Code for data preprocessing
│   ├── model_training.py  # Code for model training
│   ├── model_evaluation.py  # Code for model evaluation
│   └── data_generation.py  # Code for generating dummy data
│
├── requirements.txt  # Python package dependencies
│
└── README.md  # Project documentation and instructions
```

## Data Generation
The data_generation.py script is responsible for generating dummy product data, including product names, descriptions, prices, and categories. The generated dataset serves as the basis for training and evaluating the text classification model.

## Data Processing
The data_processing.py script preprocesses the raw data, tokenizes the text attributes, and prepares the data for model training. It also saves the preprocessed data to a pickle file for later use.

## Model Training
The model_training.py script trains a text classification model using the preprocessed data. It loads the preprocessed data, preprocesses the text data, trains the model, and saves the trained model for future use.

## Model Evaluation
The model_evaluation.py script evaluates the trained text classification model using the preprocessed data. It loads the preprocessed data, preprocesses the text data, loads the trained model, evaluates the model's performance, and saves the evaluation results to a text file.

## Usage
To use the project, follow the instructions in each script to run the data generation, data processing, model training, and model evaluation steps. Refer to the specific documentation within each script for detailed usage instructions.

## Dependencies
The project has dependencies on various Python packages. Refer to the requirements.txt file for a list of required packages and installation instructions.

## Conclusion
The text classification model demonstrates promising performance in categorizing products based on their descriptions. The project provides a foundation for further improvements and applications in product categorization and recommendation systems.

## Contributors
- Elias Assamnew
