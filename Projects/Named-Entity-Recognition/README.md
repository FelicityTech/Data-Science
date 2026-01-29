# Named Entity Recognition (NER) with Bi-LSTM Model

## Project Overview
This project implements a Named Entity Recognition (NER) system using a Bidirectional Long Short-Term Memory (Bi-LSTM) neural network. The goal is to identify and classify named entities (like persons, organizations, locations, etc.) within text data.

## Data
The dataset used for this project is sourced from `https://github.com/amankharwal/Website-data/blob/master/ner_dataset.csv?raw=true`. It contains columns for 'Sentence #', 'Word', 'POS' (Part-of-Speech Tag), and 'Tag' (Named Entity Tag).

### Data Preparation Steps:
1.  **Token and Tag Mappings**: Created mappings (`token2idx`, `idx2token`, `tag2idx`, `idx2tag`) to convert words and tags into numerical indices.
2.  **Sentence Grouping**: Grouped words and their corresponding tags by sentence.
3.  **Padding and One-Hot Encoding**: Padded sequences to a uniform length (`input_length = 104`) and converted target tags to one-hot encoded vectors.
4.  **Train-Validation-Test Split**: Divided the data into training (67.5%), validation (22.5%), and test (10%) sets.

## Model Architecture (Bi-LSTM)
The neural network model is a sequential Bi-LSTM architecture consisting of:
*   An **Embedding Layer**: Converts word indices into dense vectors.
*   A **Bidirectional LSTM Layer**: Processes sequences in both forward and backward directions to capture contextual information, with `units=64`, `dropout=0.2`, and `recurrent_dropout=0.2`.
*   Another **LSTM Layer**: Further processes the output of the Bi-LSTM layer, with `units=64`, `dropout=0.5`, and `recurrent_dropout=0.5`.
*   A **TimeDistributed Dense Layer**: Applies a Dense layer to each time step of the LSTM output to predict entity tags.

The model is compiled with `categorical_crossentropy` loss and the `adam` optimizer, with `accuracy` as the evaluation metric.

## Training
The model was trained for 25 epochs with a `batch_size=1000` and a `validation_split=0.2`.

## Evaluation
After training, the model's performance was evaluated on the test set. Key steps included:
1.  **Prediction**: Generated predictions on `test_tokens`.
2.  **Tag Conversion**: Converted one-hot encoded predictions and true `test_tags` back to their original tag indices.
3.  **Flattening**: Flattened prediction and true tag arrays for `sklearn.metrics.classification_report`.
4.  **Classification Report**: Generated a detailed classification report, including precision, recall, and F1-score for each named entity tag. The `zero_division` parameter was set to `0` to handle undefined metrics gracefully.

### Classification Report Summary:
```
              precision    recall  f1-score   support

       B-org       1.00      0.00      0.00      2038
       I-per       0.66      0.08      0.14      1715
       I-org       0.42      0.02      0.03      1697
       B-nat       0.00      0.00      0.00        17
       I-geo       0.00      0.00      0.00       681
       B-gpe       0.88      0.01      0.03      1628
       I-art       0.00      0.00      0.00        28
       B-geo       0.90      0.00      0.00      3690
       B-per       0.57      0.05      0.10      1697
       B-tim       0.00      0.00      0.00      1994
       I-nat       0.00      0.00      0.00         7
       B-art       0.00      0.00      0.00        28
           O       0.97      1.00      0.98    482811
       B-eve       0.00      0.00      0.00        34
       I-gpe       0.00      0.00      0.00        22
       I-tim       0.00      0.00      0.00       674
       I-eve       0.00      0.00      0.00        23

    accuracy                           0.97    498784
   macro avg       0.32      0.07      0.08    498784
weighted avg       0.96      0.97      0.95    498784
```

The model achieves a high overall accuracy (0.97), primarily due to the overwhelming presence of the 'O' (Outside) tag. However, the performance on specific named entity tags (e.g., 'B-org', 'I-per', 'B-geo', 'B-per') is very low, with many tags showing 0.00 precision, recall, and F1-score. This indicates the model struggles to identify and classify actual named entities other than the 'O' tag.

## Next Steps and Future Improvements
1.  **Address Class Imbalance**: The dataset is highly imbalanced, with the 'O' tag dominating. Techniques such as oversampling, undersampling, or using weighted loss functions could help improve performance on minority classes.
2.  **Hyperparameter Tuning**: Experiment with different hyperparameters for the Bi-LSTM layers (e.g., number of units, dropout rates, recurrent dropout rates) and the optimizer (learning rate, beta values).
3.  **Advanced Embeddings**: Explore pre-trained word embeddings (e.g., Word2Vec, GloVe, FastText) or contextual embeddings (e.g., BERT, ELMo) which can significantly boost NER performance.
4.  **More Complex Architectures**: Consider adding more Bi-LSTM layers, or integrating Conditional Random Fields (CRF) on top of the Bi-LSTM layer, which is a common and effective approach for sequence tagging tasks.
5.  **Feature Engineering**: Incorporate additional features like character-level embeddings for handling out-of-vocabulary words or casing information.
6.  **Qualitative Analysis**: Continue to visualize predictions on sample sentences to understand specific error patterns and guide further improvements.

## Author: [Solomon Adegoke](https://www.linkedin.com/in/solomon-eniola-adegoke/) Data Scientist - Software - ML/AI Engineering