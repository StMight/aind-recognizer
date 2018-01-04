import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    
    for x_lenght_values in test_set.get_all_Xlengths().values():
        words_probability = {}

        current_word_sequence, current_sequence =  x_lenght_values
        
        for word, model in models.items():
            try:
                score = model.score(current_word_sequence, current_sequence)
                words_probability[word] = score
            except:
                words_probability[word] = float("-inf")
                continue

        probabilities.append(words_probability)
        guesses.append(max(words_probability, key = words_probability.get))

    # return probabilities, guesses        
    return probabilities, guesses