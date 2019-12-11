import pandas as pd
import numpy as np
import re
import csv
from nltk.tokenize import WordPunctTokenizer
from tqdm import tqdm
from collections import defaultdict, Counter
from nltk.util import ngrams
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.lm import Vocabulary

n = 3

df = pd.read_csv('../data/stackexchange_800k_tokenized.csv').sample(frac = 1, random_state = 8).reset_index(drop = True)
df['tokens'] = df.tokens.apply(lambda txt : txt.split())


# build count dictionnary
'''
Using a defaultdict(Counter) structure and the nltk.utils.ngrams function
build a dictionnary of n-grams as tupels with values the counts of all following tokens

For instance for n = 3:

counts[('how', 'many')] = Counter('people': 100, 'times': 120, .... )
counts[('the', 'model')] = Counter('is': 500, 'parameters': 200, .... )

'''
counts = defaultdict(Counter)
for tokens in df.tokens.values:
    for ngram in ngrams(tokens, n= n,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>"):
        counts[ngram[:n-1]][ngram[n-1]] +=1

'''
Transform that (prefix - token) count into probability estimates (frequencies)
by normalizing each (prefix - token) count by the total number of the prefix occurence
Keep the same defaultdict(Counter)  for the probabilities
You should obtain
freq[('how', 'many')] = {'people': 0.14, 'times': 120, .... }
with p(people / how many) = c('how many people') / c('how many')
'''

# frequency
freq = defaultdict(dict)
for prefix, tokens in counts.items():
    total = sum( counts[prefix].values()  )
    for token, c in tokens.items():
        freq[prefix][token] = c / total

'''
Text generation
Given a prefix, generate the next word
- for a given prefix, sample the potential tokens with repect to their distribution
'''

text      = 'the model'
def generate(text):
    for i in range(50):
        prefix = tuple(text.split()[-n+1:])
        # no available text
        if len(freq[prefix]) == 0:
            break
        candidates  = list(freq[prefix].keys())
        probas      = list(freq[prefix].values())
        text       += ' ' + np.random.choice(candidates, p = probas)
        if text.endswith('</s>'):
            break

    return text

'''
Text generation: with temperature sampling
Given a prefix, generate the next word
- for a given prefix, sample the potential tokens with repect to their distribution
'''

def generate(text, temperature = 1, n_words=50):
    for i in range(n_words):
        prefix = tuple(text.split()[-n+1:])
        # no available next word
        if len(freq[prefix]) == 0:
            break
        candidates  = list(freq[prefix].keys())
        initial_probas = list(freq[prefix].values())
        # modify distribution
        denom   = sum( [ p ** temperature for p in initial_probas ] )
        probas  = [ p ** temperature / denom  for p in initial_probas  ]

        text       += ' ' + np.random.choice(candidates, p = probas)
        if text.endswith('</s>'):
            break

    return text


'''
Perplexity of sentence
PP(s) = pow( -1/N sum_{k+1}^{N} (  log P(w_i / w_i-k+1, ..., wi-1  )  ) , 10)
'''
sentence = "the difference between the two approaches is discussed here ."
def perplexity(sentence):
    sentence = tokenizer.tokenize(sentence.lower())
    N = len(sentence)
    logprob = 0
    for ngram in ngrams(sentence, n= n,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>"):
        logprob += np.log( freq[ ngram[:n-1]  ][ngram[n-1]]  )

    return pow(10, - 1 / N * logprob)



'''
Laplace Smoothing
When counting add a delta for existing vocab
And
- when calculating probability, if token - prefix does not exist => delta / N
'''



'''
Proba sentence
'''
def proba_sentence(sentence):
    sentence = tokenizer.tokenize(sentence.lower())
    N = len(sentence)
    logprob = 0
    for ngram in ngrams(sentence, n= n,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>"):
        try:
            logprob += np.log( freq[ ngram[:n-1]  ][ngram[n-1]]  )
        except:
            logprob += np.log( 0.000001  )
    return logprob

'''
Corpus perplexity
'''
train = df[df.type == 'answer'].reset_index()
test = df[df.type == 'title'].sample(100).text.values

for line in test:
    print(proba_sentence(line), line)






# ------------------------

train_data = [
    ngrams(t, n= n,
        pad_right=True, pad_left=True,
        left_pad_symbol="<s>", right_pad_symbol="</s>")
    for t in df.tokens]

words = [word for sent in df.tokens for word in sent]
words.extend(["<s>", "</s>"])
vocab = Vocabulary(words, unk_cutoff = 20)
model = MLE(n)
model.fit(train_data, padded_vocab)










# -------
