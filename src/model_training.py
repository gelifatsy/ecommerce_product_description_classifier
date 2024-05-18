import os
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

class TextClassifierTrainer:
    def __init__(self, data_path, text_column, label_column, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.text_column = text_column
        self.label_column = label_column
        self.test_size = test_size
        self.random_state = random_state

    def load_preprocessed_data(self):
        with open(self.data_path, 'rb') as file:
            preprocessed_data = pickle.load(file)
        self.padded_sequences = preprocessed_data['padded_sequences']
        self.input_shape = preprocessed_data['input_shape']
        self.num_classes = preprocessed_data['num_classes']
        self.vocab_size = preprocessed_data['vocab_size']
        self.preprocessed_df = preprocessed_data['preprocessed_df']

    def preprocess_text_data(self):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.preprocessed_df[self.text_column])

        train_texts, test_texts, train_labels, test_labels = train_test_split(
            self.preprocessed_df[self.text_column],
            self.preprocessed_df[self.label_column],
            test_size=self.test_size,
            random_state=self.random_state
        )

        label_encoder = LabelEncoder()
        self.train_labels_encoded = label_encoder.fit_transform(train_labels)
        self.test_labels_encoded = label_encoder.transform(test_labels)

        self.train_sequences = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=self.input_shape[0])
        self.test_sequences = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=self.input_shape[0])

    def build_model(self):
        self.model = Sequential([
            Embedding(self.vocab_size, 64, input_length=self.input_shape[0]),
            tf.keras.layers.GlobalAveragePooling1D(),
            Dense(128, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, epochs=50, batch_size=32):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = self.model.fit(
            self.train_sequences,
            self.train_labels_encoded,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.test_sequences, self.test_labels_encoded),
            callbacks=[early_stopping],
            verbose=2
        )
    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_file_path = os.path.join(save_path, 'dummy_classifier.h5')
        self.model.save(model_file_path)
        print(f"Model saved at: {model_file_path}")

def main():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed', 'preprocessed_data.pkl')
    output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'saved_model')
    
    trainer = TextClassifierTrainer(data_path, text_column='description', label_column='category')
    trainer.load_preprocessed_data()
    trainer.preprocess_text_data()
    trainer.build_model()
    trainer.train_model()
    trainer.save_model(output_directory)

if __name__ == "__main__":
    main()
