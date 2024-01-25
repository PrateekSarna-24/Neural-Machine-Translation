'''
This module covers all the functions and classes required by this project.
'''

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
import numpy as np
import pandas as pd
import contractions
from wordcloud import WordCloud

def text_to_num_sequences(text_sequences) :
    
    '''
    This function convertes text sequeces to numerical sequences and returns numerical sequences.
    '''
    
    ## create a tokenizer -> this will map each word to a number.
    ## defining the num_words -> this indicates that we need maximum of these words only to process.
    num_words = 1000
    ## initializing the tokenizer
    ## defining OOV -> this handles a word if it is out of vocabulary
    token = Tokenizer(num_words=num_words, oov_token="<UKN>")
    ## create the word_index -> word_index is the dictionary which maps the words to a numeric value.
    token.fit_on_texts(text_sequences)
    ## saving word_index.
    word_index = token.word_index
    ## limiting the words to num_words in the dictionary.
    word_indices = {word: index for word, index in token.word_index.items() if index <= num_words}
    ## converting sequences.
    num_sequences = token.texts_to_sequences(text_sequences)
    ## define vocabulary size -> this is the size of the word_index.
    ## we are incrementing it by 1 because the indexing starts with 1.
    vocab_size = len(word_indices) + 1
    
    return num_sequences, token, vocab_size, word_indices

def get_pad_sequeces(source_num_sequenecs, target_num_sequences, max_common_length = None) :
    
    '''
    This function returns padded sequences and maximun common length.
    '''
    
    ## finding out the maximum lenght of source sequences.
    source_max_len = max([len(seq) for seq in source_num_sequenecs])
    ## finding out the maximum lenght of target sequences.
    target_max_len = max([len(seq) for seq in target_num_sequences])    
    ## finding the common maximum length.
    COMMON_MAX_LENGTH = max(source_max_len, target_max_len)
    ## checking if the argument already has a max_common_length.
    if max_common_length != None :
        COMMON_MAX_LENGTH = max_common_length
    ## pad the sequences.
    source_padded_sequences = pad_sequences(source_num_sequenecs, maxlen = COMMON_MAX_LENGTH, padding='post')
    target_padded_sequences = pad_sequences(target_num_sequences, maxlen = COMMON_MAX_LENGTH, padding='post')
    
    return source_padded_sequences, target_padded_sequences, COMMON_MAX_LENGTH

def convert_data(source, target, max_common_length = None) :
    
    '''
    This function returns the complete converted set. This function calls :
    1. text_to_num_sequences -> This function convertes text sequeces to numerical sequences and returns numerical sequences.
    2. get_pad_sequeces -> This function returns padded sequences and maximun common length.S
    '''
    
    ## get numerical sequences.
    source_num_sequences, source_token, source_vocab_size, source_word_index = text_to_num_sequences(source)
    target_num_sequences, target_token, target_vocab_size, target_word_index = text_to_num_sequences(target)
    ## get padded sequences.
    source_padded_sequences, target_padded_sequences, COMMON_MAX_LENGTH = get_pad_sequeces(source_num_sequences, target_num_sequences, max_common_length = max_common_length)
    
    return source_padded_sequences, target_padded_sequences, COMMON_MAX_LENGTH, source_vocab_size, target_vocab_size, source_word_index, target_word_index

def lower_text(text) :
    '''
    Function which converts text into lower text
    '''
    return text.lower()

def expand_text(text) :
    '''
    This function expands all the words in a given sentence.
    For example it converts : 
    don't -> do not
    haven't -> have not.
    '''
    return contractions.fix(text)

def count_char(sentence) :
    '''
    This function returns the length of sentences.
    '''
    return len(sentence)

def plot_word_cloud(data):

    '''
    This function creates a wordcloud.
    '''

    ## for wordcloud to be made we need a string of all words seperated by spaces.
    words=""
    ## iterating sentences in the list of sentences.
    for sent in data:
        sent= str(sent)
        ## lowecase the sentece
        sent=sent.lower()
        ## get list of words
        tokens= sent.split()
        ## create a big string of all words in all sentences.
        words +=" ".join(tokens)+" "
    ## plot the wordcloud
    plt.figure(figsize=(15,12))
    wordcloud= WordCloud(width=800,height=400, background_color = 'white').generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')

def add_tokens(text) :
    '''
    This functions adds <START> and <END> tokens to target sentences
    '''
    return "<START> " + text + " <END>"

def predicted_translation(input_seq, encoder_model, decoder_model, target_word_index, COMMON_MAX_LENGTH):
    
    '''
    This function predicts the target output from the given input sequence.
    '''
    
    ## Predict the context vector from the encoder.
    state_values = encoder_model.predict(np.expand_dims(input_seq, axis=0)) ## coverting the 1D sequence to 2D.
    
    # Define the initial target sequence with the 'sos' token.
    target_numerical_sequence = np.zeros((1,1))
    target_numerical_sequence[0][0] = target_word_index['sos']
    
    ## define seq_len, which will work as an iterator
    seq_len = 1
    
    ## initialize the final output string
    final_translated_output = ''
    
    ## boolean variable to terminate the loop
    stop_condition = False
    
    while not stop_condition:
        
        ## Predict the next sequence and the context vector.
        output_tokens, h, c = decoder_model.predict([target_numerical_sequence] + list(state_values))
        
        # Get the predicted numerical.
        predicted_word_index = np.argmax(output_tokens[0, -1, :])
        
        # Get the word from this predicted numerical.
        predicted_word = list(target_word_index.keys())[list(target_word_index.values()).index(predicted_word_index)]
        
        ## terminating condition.
        if (predicted_word == 'eos' or seq_len > MAX_COMMON_LENGTH):
            stop_condition = True
            break
        
        ## add the word to the output string.
        target_sentence += ' ' + predicted_word
        
        # Update the target sequence by appending the predicted_word_index.
        target_numerical_sequence[0, 0] = predicted_word_index
        seq_len += 1
        
        # Update states.
        state_values = [h, c]
        
    return target_sentence

def reclean_target(sent) :
    
    '''
    This function removes 'sos' and 'eos' from the sentences.
    '''
    
    new_sent = sent.split()[1: -2]
    return ' '.join(new_sent)

def plot_bleu_scores(bleu_scores):
    '''
    This function plots the Bleu Score
    '''
    
    ## create the epochs array.
    epochs = np.arange(1, len(bleu_scores) + 1)

    # Plotting
    plt.figure(figsize=(6, 3))
    plt.plot(epochs, bleu_scores, linestyle='-', marker='o', color='b', label='BLEU Score')
    
    # Add labels and title
    plt.title('BLEU Score Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('BLEU Score')
    
    # Add grid and legend
    plt.grid(True)
    plt.legend()
    
    # Show the plot
    plt.show()

def compute_bleu(actual_target_tokens, predicted_target_tokens):

    '''
    This function computes the BLEU score.
    '''

    return sentence_bleu([actual_target_tokens], predicted_target_tokens)