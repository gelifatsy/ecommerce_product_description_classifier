import pandas as pd
import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_data(data_file_path, save_path=None):
    # Load the data
    preprocessed_df = pd.read_csv(data_file_path)

    # Tokenize the text attributes
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(preprocessed_df['description'])
    sequences = tokenizer.texts_to_sequences(preprocessed_df['description'])

    # Calculate the maximum length of the input sequences
    max_length = max(len(seq) for seq in sequences)

    # Pad the sequences to the maximum length
    padded_sequences = pad_sequences(sequences, maxlen=max_length)

    # Get the input shape and number of classes
    input_shape = (max_length,)
    num_classes = len(preprocessed_df['category'].unique())
    vocab_size = len(tokenizer.word_index) + 1

    # Save the preprocessed data if a save path is provided
    if save_path:
        preprocessed_data = {
            'padded_sequences': padded_sequences,
            'input_shape': input_shape,
            'num_classes': num_classes,
            'vocab_size': vocab_size,
            'preprocessed_df': preprocessed_df
        }
        with open(save_path, 'wb') as file:
            pickle.dump(preprocessed_data, file)

    return padded_sequences, input_shape, num_classes, vocab_size, preprocessed_df

if __name__ == '__main__':
    # Specify the path to the preprocessed data file
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'generated')
    csv_file_path = os.path.join(directory, 'dummy_product_data.csv')
    output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')
    save_path = os.path.join(output_directory, 'preprocessed_data.pkl')

    # Preprocess the data
        # Preprocess the data and save it
    preprocess_data(csv_file_path, save_path=save_path)

    print(f"Preprocessed data saved to: {save_path}")
