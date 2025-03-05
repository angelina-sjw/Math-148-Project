import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input, Concatenate
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                            TrainingArguments, Trainer)

### BERT Fine-Tuning Functions ###
def finetune_bert(train_dataset, test_dataset, model_path, output_dir):
    """
    Fine-tunes a BERT model for binary text classification.

    Args:
        train_dataset: The training dataset (compatible with Hugging Face Datasets).
        test_dataset: The evaluation dataset (compatible with Hugging Face Datasets).
        model_path: The path to the pretrained BERT model.
        output_dir: The directory to save checkpoints and logging information.

    Returns:
        A Hugging Face Trainer object after training and evaluation.
    """
    # Load tokenizer from the pretrained model
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Define label mappings for a binary classification task
    id2label = {0: "less useful", 1: "more useful"}
    label2id = {"less useful": 0, "more useful": 1}

    # Initialize a BERT model with the given label configuration
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=2, 
        id2label=id2label, 
        label2id=label2id
    )

    # Set up training arguments for the Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir='./bert_logs',
        load_best_model_at_end=True,
        metric_for_best_model='f1'
    )
    
    # Define a custom metric function for the Trainer
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        acc = accuracy_score(labels, predictions)
        return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}
    
    # Create the Trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluate the model on the test set
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

    return trainer

### LSTM Model for Text ###
def build_lstm_model(max_words, max_length, embedding_dim=64):
    """
    Builds a simple LSTM model for binary text classification.

    Args:
        max_words: The maximum vocabulary size for the embedding layer.
        max_length: The maximum sequence length of the input data.
        embedding_dim: The dimensionality of the embedding layer.

    Returns:
        A compiled Keras Sequential model.
    """
    # Create a Sequential model with Embedding, LSTM, and Dense layers
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_length),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model with an optimizer, loss function, and evaluation metric
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

### Multi-Input Model (Text + Other Features) ###
def build_multi_input_model(max_length, max_words, num_other_features, embedding_dim=64):
    """
    Builds a multi-input Keras model for combining text features (via LSTM) 
    and additional numeric features.

    Args:
        max_length: The maximum length of the input text sequences.
        max_words: The maximum vocabulary size for the embedding layer.
        num_other_features: The dimensionality of the additional feature vector.
        embedding_dim: The dimensionality of the embedding layer.

    Returns:
        A compiled Keras Model object.
    """
    # Define text input branch
    text_input = Input(shape=(max_length,), name='text_input')
    x = Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_length)(text_input)
    x = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(x)
    x = Dense(32, activation='relu')(x)

    # Define the other (numeric) features input branch
    other_input = Input(shape=(num_other_features,), name='other_input')
    y_branch = Dense(32, activation='relu')(other_input)

    # Concatenate both branches
    combined = Concatenate()([x, y_branch])
    combined = Dense(32, activation='relu')(combined)
    combined = Dropout(0.2)(combined)
    output = Dense(1, activation='sigmoid', name='output')(combined)
    
    # Build and compile the final multi-input model
    model = Model(inputs=[text_input, other_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model