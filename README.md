# Deep Learning for Social Media Post Classification

## Overview
This project is part of the Deep Learning assignments focusing on classifying social media posts into real information or fake misinformation categories. The model leverages datasets from Constraint@AAAI-2021, targeting COVID-19 related misinformation spread across platforms like Twitter, Facebook, and Instagram.

## Dataset
The dataset encompasses 10,600 samples sourced from various social media platforms, labeled as either 'real' (5,545 samples) or 'fake' (5,055 samples). For more details, visit [Constraint@AAAI-2021](<Link to your dataset>).

## Models
We experimented with three types of deep learning models:
- Deep Neural Network (DNN)
- 1D Convolutional Neural Network (1D-CNN)
- AutoModelForSequenceClassification

## Dependencies
- Python 3.6+
- PyTorch
- Transformers
- Scikit-learn
- SciPy
- Numpy
- Pandas

To install the required libraries, run:
```bash
pip install torch transformers scikit-learn scipy numpy pandas
## Usage
Follow these steps to preprocess data, train models, and evaluate results. Ensure you are in the project's root directory before running these commands.

1. **Preprocessing:** Convert the raw social media posts into a format suitable for training. This step involves cleaning the text and preparing it for vectorization.
   ```bash
   python Preprocess.py
2. ** Vectorization **
Generate vector representations of the preprocessed social media posts using one of the specified BERT models. Replace `<bert-model>` with your chosen model.
```bash
python Vectorize.py --model_name <bert-model>

3. ** Training Models **
Train the Deep Neural Network (DNN), 1D Convolutional Neural Network (1D-CNN), and AutoModelForSequenceClassification models using the vectorized data.
python DNN.py
python CNN.py
python AutoModel.py

4. ** Evaluation **
Evaluate the trained models on the test set. Provide the path to the saved model when prompted. Replace <path-to-saved-model> with the actual path to your model.
python RunEval.py --model_path <path-to-saved-model>

5. ** Results **
The evaluation metrics include classification accuracy, F1-score, precision, and recall, both micro and macro. The results are detailed in the project report, highlighting the performance of each model on the test set.

6. ** Model Checkpoints **
Access the trained model checkpoints via the provided Google Drive link. Each model's best checkpoint, as determined by validation performance, is saved for further analysis and inference.

7. ** Running End-to-Endn **
For convenience, a shell script runEndtoEnd.sh is provided to execute all steps sequentially, from data preprocessing to evaluation.
To run, execute the following command in the terminal:
bash runEndtoEnd.sh




