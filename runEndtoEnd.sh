#!/bin/bash

# Check if exactly 3 arguments are passed
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <data-excel-filepath> <bertmodelname> <modelName (DNN or 1D-CNN)>"
    exit 1
fi

# Assigning command line arguments to variables
DATA_PATH="$1"
BERT_MODEL_NAME="$2"
MODEL_NAME="$3"

# Step 1: Preprocess the data
echo "Preprocessing data from $DATA_PATH..."
python3 preprocess.py "$DATA_PATH"

# Step 2: Vectorize the data
echo "Vectorizing data using BERT model $BERT_MODEL_NAME..."
python3 vectorization.py "$BERT_MODEL_NAME"

# Steps 3 and 4: Train the model
# It seems like step 4 was a repeat of step 3. If you intended a different command for step 4, adjust accordingly.
echo "Training model $MODEL_NAME using vectorized data..."
python3 "$MODEL_NAME.py" "XtrainVectorized${BERT_MODEL_NAME}.pt"

# Step 5: Run evaluation
echo "Running evaluation for $MODEL_NAME model..."
python3 runEval.py "$MODEL_NAME" "${MODEL_NAME}_${BERT_MODEL_NAME}.pth"

echo "Pipeline execution completed."
