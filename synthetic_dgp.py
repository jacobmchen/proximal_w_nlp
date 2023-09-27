"""
Code for running a semi-synthetic DGP setup to find average causal effect and confidence intervals.
"""

import pandas as pd
import numpy as np
from scipy.special import expit
import statsmodels.api as sm
import scipy.stats as stats
from adjustment import *
from gensim.models import Word2Vec
import pickle

from proximal import *
from regex_predictor import *
from odds_ratio import *
from backdoor import *
from bag_of_words import *
from cohens_kappa import *

def create_semi_synthetic_dataframe(oracle, W, Z, causal_effect=1.3, seed=1):
    master_data = pd.read_csv('csv_files/master_data.csv')

    semi_synthetic_data = pd.DataFrame({'U': master_data[oracle], 'W': W, 'Z': Z,
                                    'age': master_data['age'], 'gender': master_data['gender']})
        
    # generate semi-synthetic data
    np.random.seed(seed)

    size = len(semi_synthetic_data)

    C = np.random.normal(76.4, 56.8, size)

    # age is a fairly large continuous variable, so it is necessary to make it smaller
    A = np.random.binomial(1, expit(0.8*semi_synthetic_data['U'] + 0.8*semi_synthetic_data['gender'] + 0.8*(semi_synthetic_data['age'] - 67)), size)
    # A = np.random.binomial(1, expit(0.8*semi_synthetic_data['U'] + 0.8*semi_synthetic_data['gender']), size)

    Y = np.random.normal(0, 1, size) + causal_effect*A + 1.4*semi_synthetic_data['U'] + 0.8*semi_synthetic_data['gender'] + 0.5*semi_synthetic_data['age']
    # Y = np.random.normal(0, 1, size) + 1.3*A + 1.4*semi_synthetic_data['U'] + 0.8*semi_synthetic_data['gender']

    semi_synthetic_data['A'] = A
    semi_synthetic_data['Y'] = Y
    semi_synthetic_data['C'] = C

    return semi_synthetic_data

def run_semi_synthetic_dgp(oracle='afib', classifier='document', verbose=False):
    # check if inputs are valid
    if oracle != 'afib' and oracle != 'heart_fail' and oracle != 'kidney_fail':
        return 'invalid input for oracle'
    if classifier != 'sentence' and classifier != 'document':
        return 'invalid input for classifier'

    '''
    Main code portion:
    '''
    master_data = pd.read_csv('csv_files/master_data.csv')

    # find the candidates for using regular expression matching
    regex_candidates = []

    if oracle == 'afib':
        if classifier == 'sentence':
            zero_shot_preds = pd.read_csv('csv_files/predictions-xxl-atrialfibrillation-sentence.csv')
        elif classifier == 'document':
            zero_shot_preds = pd.read_csv('csv_files/predictions-xxl.csv')
        regex_candidates.append('atrial')
        regex_candidates.append('fibrillation')
    elif oracle == 'heart_fail':
        if classifier == 'sentence':
            zero_shot_preds = pd.read_csv('csv_files/predictions-xxl-congestiveheartfailure-sentence.csv')
        elif classifier == 'document':
            zero_shot_preds = pd.read_csv('csv_files/predictions-xxl-congestiveheartfailure-document.csv')
        regex_candidates.append('congestive')
        regex_candidates.append('heart')
        regex_candidates.append('failure')
    elif oracle == 'kidney_fail':
        if classifier == 'sentence':
            zero_shot_preds = pd.read_csv('csv_files/predictions-xxl-acutekidneyfailure-sentence.csv')
        elif classifier == 'document':
            zero_shot_preds = pd.read_csv('csv_files/predictions-xxl-acutekidneyfailure-document.csv')
        regex_candidates.append('acute')
        regex_candidates.append('kidney')
        regex_candidates.append('failure')

    # train a Word2Vec model that can tell you the nearest neighbors for a certain word
    model = Word2Vec.load('word2vec.model')
    similar_words = model.wv.most_similar(positive=regex_candidates)

    for i in range(5):
        regex_candidates.append(similar_words[i][0])

    if verbose:
        print(regex_candidates)

    ace_predictions = []
    conf_intervals = []
    concurrency = []
    for word in regex_candidates:
        regex_preds = regular_expression_predict(master_data['notes_half2'], [word])

        semi_synthetic_data = create_semi_synthetic_dataframe(oracle, zero_shot_preds['prediction'], regex_preds)

        if verbose:
        #     print(odds_ratio('U', 'W', [], semi_synthetic_data))
        #     print(odds_ratio('U', 'Z', [], semi_synthetic_data))

        #     print()
            # concurrency.append(np.mean(semi_synthetic_data['W'] == semi_synthetic_data['Z']))
            concurrency.append(cohens_kappa(semi_synthetic_data['W'], semi_synthetic_data['Z']))
        #     print()

            # odds_ratio.append(odds_ratio('W', 'Z', ['U'], semi_synthetic_data))
        #     print(odds_ratio('W', 'Z', ['U', 'age', 'gender'], semi_synthetic_data))

        # approximate the ACE
        ace_predictions.append(proximal_find_ace('A', 'Y', 'W', 'Z', ['age', 'gender'], semi_synthetic_data))
        conf_intervals.append(compute_confidence_intervals('A', 'Y', 'W', 'Z', ['age', 'gender'], semi_synthetic_data))

    return (ace_predictions, conf_intervals, concurrency)

def run_causal_null_hypothesis(oracle='afib', classifier='document', causal_effect=0, num_bootstraps=200, alpha=0.05, verbose=False, sample_size=None):
    # check if inputs are valid
    if oracle != 'afib' and oracle != 'heart_fail' and oracle != 'kidney_fail':
        return 'invalid input for oracle'
    if classifier != 'sentence' and classifier != 'document':
        return 'invalid input for classifier'
    
    master_data = pd.read_csv('csv_files/master_data.csv')

    # find the candidates for using regular expression matching
    regex_candidates = []

    if oracle == 'afib':
        if classifier == 'sentence':
            zero_shot_preds = pd.read_csv('csv_files/predictions-xxl-atrialfibrillation-sentence.csv')
        elif classifier == 'document':
            zero_shot_preds = pd.read_csv('csv_files/predictions-xxl.csv')
        regex_candidates.append('atrial')
        regex_candidates.append('fibrillation')
    elif oracle == 'heart_fail':
        if classifier == 'sentence':
            zero_shot_preds = pd.read_csv('csv_files/predictions-xxl-congestiveheartfailure-sentence.csv')
        elif classifier == 'document':
            zero_shot_preds = pd.read_csv('csv_files/predictions-xxl-congestiveheartfailure-document.csv')
        regex_candidates.append('congestive')
        regex_candidates.append('heart')
        regex_candidates.append('failure')
    elif oracle == 'kidney_fail':
        if classifier == 'sentence':
            zero_shot_preds = pd.read_csv('csv_files/predictions-xxl-acutekidneyfailure-sentence.csv')
        elif classifier == 'document':
            zero_shot_preds = pd.read_csv('csv_files/predictions-xxl-acutekidneyfailure-document.csv')
        regex_candidates.append('acute')
        regex_candidates.append('kidney')
        regex_candidates.append('failure')

    # train a Word2Vec model that can tell you the nearest neighbors for a certain word
    model = Word2Vec.load('word2vec.model')
    similar_words = model.wv.most_similar(positive=regex_candidates)

    for i in range(5):
        regex_candidates.append(similar_words[i][0])
    
    if verbose:
        print(regex_candidates)

    datasets = []

    for word in regex_candidates:
        regex_preds = regular_expression_predict(master_data['notes_half2'], [word])
        semi_synthetic_data = create_semi_synthetic_dataframe(oracle, zero_shot_preds['prediction'], regex_preds, causal_effect=causal_effect)
        
        # pd.DataFrame({'U': master_data[oracle], 'W': zero_shot_preds['prediction'], 'Z': regex_preds,
        #                            'age': master_data['age'], 'gender': master_data['gender']})
        
        # control the sample size of the experiment if desired
        if sample_size != None:
            semi_synthetic_data = semi_synthetic_data.sample(sample_size, replace=False)
            semi_synthetic_data.reset_index(drop=True, inplace=True)

        datasets.append(semi_synthetic_data)

    # bootstrap the product of the causal effects
    Ql = alpha/2
    Qu = 1 - alpha/2
    products = []
    for i in range(num_bootstraps):
        product = 1

        for data in datasets:
            # resample the data with replacement
            data_sampled = data.sample(len(data), replace=True)
            data_sampled.reset_index(drop=True, inplace=True)
            product = product * proximal_find_ace('A', 'Y', 'W', 'Z', ['age', 'gender'], data_sampled)
        
        if verbose:
            print(product)

        products.append(product)

    # calculate the quantiles
    quantiles = np.quantile(products, q=[Ql, Qu])
    q_low = quantiles[0]
    q_up = quantiles[1]
    
    return q_low, q_up, products

def run_backdoor_adjustment(oracle='afib', classifier='document', causal_effect=0):
    # run backdoor adjustment using the zero-shot predictors as the backdoor adjustment set

    # check if inputs are valid
    if oracle != 'afib' and oracle != 'heart_fail' and oracle != 'kidney_fail':
        return 'invalid input for oracle'
    if classifier != 'sentence' and classifier != 'document':
        return 'invalid input for classifier'
    
    master_data = pd.read_csv('csv_files/master_data.csv')

    if oracle == 'afib':
        if classifier == 'sentence':
            zero_shot_preds = pd.read_csv('csv_files/predictions-xxl-atrialfibrillation-sentence.csv')
        elif classifier == 'document':
            zero_shot_preds = pd.read_csv('csv_files/predictions-xxl.csv')
    elif oracle == 'heart_fail':
        if classifier == 'sentence':
            zero_shot_preds = pd.read_csv('csv_files/predictions-xxl-congestiveheartfailure-sentence.csv')
        elif classifier == 'document':
            zero_shot_preds = pd.read_csv('csv_files/predictions-xxl-congestiveheartfailure-document.csv')
    elif oracle == 'kidney_fail':
        if classifier == 'sentence':
            zero_shot_preds = pd.read_csv('csv_files/predictions-xxl-acutekidneyfailure-sentence.csv')
        elif classifier == 'document':
            zero_shot_preds = pd.read_csv('csv_files/predictions-xxl-acutekidneyfailure-document.csv')

    # create the semi-synthetic dataset with just the zero-shot predictions as both of the proxies
    semi_synthetic_data = create_semi_synthetic_dataframe(oracle, zero_shot_preds['prediction'], zero_shot_preds['prediction'], causal_effect=causal_effect)

    point_estimate = backdoor_adjustment('Y', 'A', ['W', 'age', 'gender'], semi_synthetic_data)
    confidence_interval = compute_confidence_intervals_backdoor('Y', 'A', ['W', 'age', 'gender'], semi_synthetic_data, 'backdoor')

    return point_estimate, confidence_interval


def evaluate_errors(oracle='afib', sample_sizes=[100, 500, 1000, 1500], num_iterations=100, verbose=False):
    results = []

    for sample_size in sample_sizes:
        
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for i in range(num_iterations):
            interval = run_causal_null_hypothesis(oracle=oracle, causal_effect=0, sample_size=sample_size)
            if verbose:
                print(interval)

            if interval[0] < 0 and interval[1] > 0:
                tn += 1
            else:
                fp += 1

            interval = run_causal_null_hypothesis(oracle=oracle, causal_effect=1.3, sample_size=sample_size)
            if verbose:
                print(interval)
                
            if interval[0] < 0 and interval[1] > 0:
                fn += 1
            else:
                tp += 1

        results.append((fp/(tn+fp), fn/(tp+fn)))

    return results
