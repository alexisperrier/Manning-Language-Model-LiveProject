'''
Part II: n-gram language model
see yandex
'''


import pandas as pd
import numpy as np
import re
import csv
from nltk.tokenize import WordPunctTokenizer
from tqdm import tqdm
from collections import defaultdict, Counter
from nltk.util import ngrams

df = pd.read_csv('../data/fulltext_71k.csv')

'''
Convert texts into string of space separated tokens
'''

tokenizer = WordPunctTokenizer()
df['tokens'] = df.fulltext.apply(lambda t : ' '.join(tokenizer.tokenize(t.lower())) )

'''
Write a function that takes the string of tokens as input, and the N in n-grams
and outputs
for each n-gram: a Counter of follow up words
for instance:

counts[(how', 'many')] = Counter('people': 100, 'times': 120, .... )
use the ngrams function from nltk.util
add EOS and UNK as end and beginning of sentence

- alternative structures ?
- merge with proba?

'''


# special tokens:
# - unk represents absent tokens,
# - eos is a special token after the end of sequence

UNK, EOS = "_UNK_", "_EOS_"

def count_ngrams(lines, n):
    counts = defaultdict(Counter)
    for line in sorted(lines, key=len):
        for gram in ngrams(line.split(),
                        n = n,
                        pad_right=True,
                        pad_left=True,
                        left_pad_symbol = UNK,
                        right_pad_symbol = EOS):
            counts[gram[:n-1]][ gram[n-1] ] += 1

    return counts

#
counts = count_ngrams(df.tokens, 3)

def prefix_word_proba(counts):
    '''
    returns the probability (MLE) for each prefix -> token
    p(token / prefix)
    '''
    prob = defaultdict(Counter)
    logprob = defaultdict(Counter)
    for prefix, nums in counts.items():
        total = sum( nums.values())
        for w,k in nums.items():
            prob[prefix][w] = k / total
            logprob[prefix][w] = np.log( (k / total) + 1)

    return prob, logprob

def get_possible_next_tokens(prefix):
    """
    :param prefix: string with space-separated prefix tokens
    :returns: a dictionary {token : it's probability} for all tokens with positive probabilities
    """
    prefix = prefix.split()
    prefix = prefix[max(0, len(prefix) - n + 1):]
    prefix = [ UNK ] * (n - 1 - len(prefix)) + prefix
    return prob[tuple(prefix)]

'''
Add temperature to select next token
'''

def get_next_token(prefix, temperature=1.0):
    """
    return next token after prefix;
    :param temperature: samples proportionally to lm probabilities ^ temperature
        if temperature == 0, always takes most likely token. Break ties arbitrarily.
    """
    next_tks = get_possible_next_tokens(prefix)

    denom = sum( [ p ** temperature for p in  next_tks.values() ] )

    candidates, probas = [], []
    for tok, proba in next_tks.items():
        candidates.append(tok)
        probas.append(proba ** temperature / denom )

    return np.random.choice(candidates, p = probas)

'''
Play: generate sentences from seed
'''

prefix = 'thanks' # <- your ideas :)

for i in range(100):
    prefix += ' ' + get_next_token(prefix, 0.8)
    if prefix.endswith(EOS) or len(get_possible_next_tokens(prefix)) == 0:
        break

print(prefix)

'''
Calculate probability of a few sentences
'''


'''
Calculate perplexity
https://stats.stackexchange.com/questions/129352/how-to-find-the-perplexity-of-a-corpus
'''






'''
Change the initial dataset
* sentence based
* rm $$, numbers, some punctuations : | () {}, ... but keep ,.?!
*
'''



# ----
