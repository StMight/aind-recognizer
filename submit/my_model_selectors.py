import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection based on BIC scores
        current_best_score = float('inf')
        current_best_model = None

        # the number of observations, or equivalently, the sample size
        rows, columns = self.X.shape

        for state_number in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(state_number)
                score = model.score(self.X, self.lengths)

                number_of_parameters =  (state_number * 2) + (2 * columns * state_number) - 1

                #bic_score = -2 * np.log(score) + number_of_parameters * np.log(rows)
                bic_score = -2 * score + number_of_parameters * np.log(rows)

            except:
                bic_score = float('inf')

            if(bic_score < current_best_score):
                    current_best_score = bic_score
                    current_best_model = model

        if(current_best_model == None or current_best_score == float('inf')):
            return self.base_model(self.n_constant)
        else:
            return current_best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        current_best_score = float('-inf')
        current_best_model = None

        scores = {}

        for state_number in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(state_number)

                score = model.score(self.X, self.lengths)
                scores[state_number] = score
            except:
                pass

        # M should be the number of given categories
        M = self.max_n_components - self.min_n_components + 1

        for state_number in scores:
            # DIC = log(P(X(i)) - 1/(M - 1) * sum(log(P(X(all but i))
            
            score = scores[state_number]
            score_sum = np.sum([scores[state_number_sum] for state_number_sum in scores if state_number != state_number_sum])
            
            dic_score = score - 1 / (M - 1) - score_sum

            if(dic_score > current_best_score):
                current_best_score = dic_score
                current_best_model = self.base_model(state_number)     

        return current_best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

        # if len(self.sequences) >= 2:
        kf = KFold(n_splits = 3, shuffle = False, random_state = None)
        # else:
        #     return self.base_model(self.n_constant)

        current_best_score = float('-inf')
        current_best_model = None

        for state_number in range(self.min_n_components, self.max_n_components + 1):
            cv_score = []
            
            try:
                if(len(self.sequences) > 2):
                    for train_index, test_index in kf.split(self.sequences):
                        
                        # Training sequences recombined
                        self.X, self.lengths = combine_sequences(train_index, self.sequences)

                        x_test, lengths_test = combine_sequences(test_index, self.sequences)

                        current_model = self.base_model(state_number)
                        cv_score.append(current_model.score(x_test, lengths_test))
                else:
                    current_model = self.base_model(state_number)
                    cv_score.append(current_model.score(self.X, self.lengths))

                current_score = np.mean(cv_score)
            except:
                continue

            if(current_score > current_best_score):
                current_best_score = current_score
                current_best_model = self.base_model(state_number)

        return current_best_model