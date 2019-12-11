import pandas as pd
import numpy as np
import re
import csv

df = pd.read_csv('../data/StatsExchange.csv')

'''
Flatten dataset
'''

df.rename(columns = { 'Title': 'title', 'Body': 'body', 'Text': 'text'}, inplace = True)

# title text and Body as separate items

title = df.title.unique()
body =  df.body.unique()
text =  df.text.unique()

data = pd.DataFrame(np.concatenate((text, body, title)), columns = ['fulltext']).dropna()

'''
Alternative data aggregation
Aggregate title + Body + multiple text
'''
# aggregate text by posts
post = df.groupby(by = 'PostId')['text'].apply(lambda x: ','.join(x)).reset_index()
# merge unique title / body with aggregated texts
data = df.drop_duplicates(subset = 'PostId')[['PostId','title', 'body']].merge(post, on = 'PostId'  )

data.fillna('',inplace = True)

data['fulltext'] = data.apply(lambda d: ' '.join([d.title, d.body, d.text])   , axis = 1)



# remove html tags
# ex: text = re.sub("<[^>]*>",' ', text)

data['fulltext'] = data.fulltext.apply(lambda t : re.sub("<[^>]*>",' ', t) )
data['fulltext'] = data.fulltext.apply(lambda t : re.sub("[\r\n]+",' ', t) )
data['fulltext'] = data.fulltext.apply(lambda t : re.sub("http\S+",' ', t) )
data['fulltext'] = data.fulltext.apply(lambda t : t.strip() )

data['len_text'] = data.fulltext.apply(len)
data.sort_values(by = 'len_text', inplace = True)

data = data[data.len_text > 2]
data.to_csv('../data/fulltext_71k.csv', quoting = csv.QUOTE_ALL, index = False)
# remove latex : $\\overline{x}\\pm \\sigma Z_{\\alpha/2} $

# remove urls

# remove mentions @
