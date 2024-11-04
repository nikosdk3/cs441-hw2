# Spark DL4J LLM Project

## Overview
This project implements a large language model (LLM) training pipeline using Apache Spark and DeepLearning4j (DL4J). It processes text data to build a vocabulary, trains an LSTM model on sequences of tokenized input, and evaluates its performance.

## Components
- **DataLoader**: Loads and processes input data, creates sliding windows for training.
- **Model**: Defines the neural network architecture, including the LSTM and output layers.
- **Main**: Orchestrates the Spark session, data loading, and model training.
- **Config**: Configuration class for retrieving config values.
- **Utils**: Helper functions used between multiple classes.
- **ModelTest**: Testing model generation by predicting words based on seed text.

## Prerequisites
All dependencies used are included in the build.sbt.

## Build and Run

1. **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Build the project using SBT**:
    ```bash
    sbt clean compile
    ```

3. **Run the project**:
    ```bash
    sbt run
    ```

### Data Partitioning
The input data is partitioned based on sequence length and stride parameters. The `DataLoader` class creates overlapping sequences from the tokenized input, generating batches for training. 

### Input/Output Specifications
- **Input**: Text file containing the raw text data (`the-verdict.txt`).
- **Output**: Model training logs, statistics, and saved model weights after training completion.

## Performance Metrics
During training, the following metrics are logged:
- **Training Loss**: Cross-entropy loss to measure model performance.
- **Training Accuracy**: Percentage of correctly predicted tokens.
- **Gradient Statistics**: Norms to monitor updates.
- **Learning Rate**: Current learning rate and its decay.
- **Time per Epoch**: Duration of each training epoch.
- **Time per Iteration**: Duration of each batch evaluation.
- **Batch Processing Time**: How long it takes for the model to train on a batch.

## Monitoring and Debugging
- Use Spark UI to monitor execution times, executor metrics, and task statistics.

Here are some statistics you can view in the Spark UI:
- **Memory Usage**: RAM usage during training.
- **Data Shuffling Statistics**: Amount of data shuffled across Spark executors.
- **CPU/GPU Utilization**: Resource usage statistics during training.
