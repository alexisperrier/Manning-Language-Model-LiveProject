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
n  = 3
counts = count_ngrams(df.tokens, n)

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

prob, logprob = prefix_word_proba(counts)

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
sent  = df.sample().tokens.values[0]

def perplexity(sent):
    pp = 1
    lpp = 0
    for gram in ngrams(sent.split(),
                    n = n,
                    pad_right=True,
                    pad_left=True,
                    left_pad_symbol = UNK,
                    right_pad_symbol = EOS):
        prefix = tuple(gram[:n-1])
        word = gram[n-1]
        p = prob[prefix][word]
        lpp += np.log(p)
        print(gram, prefix, word, p, np.log(p), lpp)
    print()
    print( - 1/n *  lpp)




import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.lm import Vocabulary

train_sentences = ['an apple', 'an orange']
train_sentences = ['this is the day', 'this day is great','great you have the day']
tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in train_sentences]

n = 2
train_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
words = [word for sent in tokenized_text for word in sent]
words.extend(["<s>", "</s>"])
padded_vocab = Vocabulary(words)
model = MLE(n)
model.fit(train_data, padded_vocab)

test_sentences = ['an apple', 'an ant']

test_sentences = ['the day', 'this is','this day']
tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in test_sentences]

test_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
for test in test_data:
    print ("MLE Estimates:", [((ngram[-1], ngram[:-1]),model.score(ngram[-1], ngram[:-1])) for ngram in test])

test_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
for i, test in enumerate(test_data):
  print("PP({0}):{1}".format(test_sentences[i], model.perplexity(test)))


'''
Calculate perplexity
https://stats.stackexchange.com/questions/129352/how-to-find-the-perplexity-of-a-corpus
'''


# perplexity of a sentence
sent = ""



'''
Change the initial dataset
* sentence based
* rm $$, numbers, some punctuations : | () {}, ... but keep ,.?!
*
'''



# ----
