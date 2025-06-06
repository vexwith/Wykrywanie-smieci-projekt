import tensorflow as tf
from keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, Dropout
from keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np


class GarbageClassifier:
    def __init__(self, input_labels, output_categories=None,
                 vocab_size=1000, max_len=20, embedding_dim=16,
                 hidden_layers=[64, 32], dropout_rate=0.2):
        """
        Initialize the garbage classification model.

        Args:
            input_labels: List of input garbage labels (strings)
            output_categories: List of output categories (default: ["plastic", "paper", "bio", "glass", "mixed"])
            vocab_size: Size of vocabulary for tokenizer
            max_len: Maximum length of input sequences
            embedding_dim: Dimension for embedding layer
            hidden_layers: List of integers specifying units in hidden layers
            dropout_rate: Dropout rate for regularization
        """
        if output_categories is None:
            self.output_categories = ["plastic", "paper", "bio", "glass", "mixed"]
        else:
            self.output_categories = output_categories

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate

        # Initialize tokenizer
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(input_labels)

        # Build the model
        self.model = self._build_model()

    def _build_model(self):
        """Build the neural network model with configurable layers."""
        # Input layer
        inputs = Input(shape=(self.max_len,))

        # Embedding layer
        x = Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_len)(inputs)
        x = GlobalAveragePooling1D()(x)

        # Hidden layers
        for units in self.hidden_layers:
            x = Dense(units, activation='relu')(x)
            x = Dropout(self.dropout_rate)(x)

        # Output layer
        outputs = Dense(len(self.output_categories), activation='softmax')(x)

        # Create model
        model = Model(inputs=inputs, outputs=outputs)

        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def preprocess_text(self, text_labels):
        """Convert text labels to padded sequences."""
        sequences = self.tokenizer.texts_to_sequences(text_labels)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        return padded

    def train(self, X_train, y_train, epochs=10, batch_size=32, validation_data=None):
        """
        Train the model.

        Args:
            X_train: Training text data (list of strings)
            y_train: Training labels (indices corresponding to output_categories)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_data: Optional validation data tuple (X_val, y_val)
        """
        # Preprocess text data
        X_train_seq = self.preprocess_text(X_train)

        # Convert labels to indices if they're strings
        if isinstance(y_train[0], str):
            y_train = np.array([self.output_categories.index(label) for label in y_train])

        # Train the model
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_seq = self.preprocess_text(X_val)
            if isinstance(y_val[0], str):
                y_val = np.array([self.output_categories.index(label) for label in y_val])
            validation_data = (X_val_seq, y_val)

        history = self.model.fit(
            X_train_seq, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data
        )

        return history

    def predict(self, text_labels):
        """Predict categories for new text labels."""
        sequences = self.preprocess_text(text_labels)
        predictions = self.model.predict(sequences)
        predicted_indices = np.argmax(predictions, axis=1)
        return [self.output_categories[i] for i in predicted_indices]

    def summary(self):
        """Print model summary."""
        return self.model.summary()


# Example usage
if __name__ == "__main__":
    # Example training data
    input_labels = [
        "water bottle", "newspaper", "banana peel", "beer bottle", "pizza box",
        "plastic bag", "cardboard", "apple core", "wine bottle", "chip bag",
        "yogurt container", "magazine", "orange peel", "jam jar", "diaper",
        "shampoo bottle", "notebook", "coffee grounds", "perfume bottle", "tetrapak"
    ]

    # Corresponding output categories (must match one of the 5 main types)
    output_labels = [
        "plastic", "paper", "bio", "glass", "mixed",
        "plastic", "paper", "bio", "glass", "mixed",
        "plastic", "paper", "bio", "glass", "mixed",
        "plastic", "paper", "bio", "glass", "mixed"
    ]

    # Initialize classifier with configurable parameters
    classifier = GarbageClassifier(
        input_labels=input_labels,
        hidden_layers=[64, 32],  # Two hidden layers with 64 and 32 units respectively
        dropout_rate=0.3,
        vocab_size=500,
        max_len=10,
        embedding_dim=32
    )

    # Print model summary
    classifier.summary()

    # Train the model
    history = classifier.train(
        X_train=input_labels,
        y_train=output_labels,
        epochs=20,
        batch_size=4
    )

    # Test the model
    test_labels = ["soda can", "milk carton", "egg shell", "plastic wrap"]
    predictions = classifier.predict(test_labels)
    for label, pred in zip(test_labels, predictions):
        print(f"{label}: predicted as {pred}")