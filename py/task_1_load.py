'''
Task 1: load and cleanup dataset
- rm html tags
- rm line returns
- rm urls
- rm punctuation except ,.?!
- rm latex
- tokenize
- remove text with less than 3 tokens

'''

import pandas as pd
import re
import string
import csv

data = pd.read_csv('../data/stackexchange_800k.csv')


'''
Clean up with regex
'''
# rm html tags
data['text'] = data.text.apply(lambda t : re.sub("<[^>]*>",' ', t) )
# rm line returns
data['text'] = data.text.apply(lambda t : re.sub("[\r\n]+",' ', t) )
# rm urls
data['text'] = data.text.apply(lambda t : re.sub("http\S+",' ', t) )
# rm mentions
data['text'] = data.text.apply(lambda t : re.sub("@\S+",' ', t) )
# rm latex
data['text'] = data.text.apply(lambda t : re.sub("\$[^>]*\$",' ', t) )
# rm digits
data['text'] = data.text.apply(lambda t : re.sub("\d+",' ', t) )
# rm punctuation
remove = '"#$%&()*+/:;<=>@[\\]^_`{|}~”“'
pattern = r"[{}]".format(remove)
data['text'] = data.text.apply(lambda t : re.sub(pattern,' ', t) )
# rm digits

# rm multiple spaces
data['text'] = data.text.apply(lambda t : re.sub("\s\s+",' ', t) )

# strip
data['text'] = data.text.apply(lambda t : t.strip() )



'''
Tokenize
Tokenize the text using a standard tokenizer (nltk, spacy, ...)
Feel free to reduce the corpus by removing items that have only a few tokens (1, 2, 3,...)
'''

from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()
data['tokens'] = data.text.apply(lambda t : ' '.join(tokenizer.tokenize(t.lower())) )

'''
rm rows with fewer than 4 tokens
'''
data['n_tokens'] = data.tokens.apply(len)

data = data[(data.n_tokens > 3) & (data.n_tokens < 5000)].reset_index(drop = True)

data.to_csv('../data/stackexchange_800k_tokenized.csv', quoting = csv.QUOTE_ALL, index = False)

# data = pd.read_csv('../data/stackexchange_800k_tokenized.csv')
# data['tokens'] = data.tokens.apply(lambda text : text.split())


# -----------------
