import pandas as pd
import numpy as np
import re
import csv
import os


'''
Read files from data/QueryResults_XY.csv and aggregate
'''

def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

df = pd.DataFrame()
path = "../data/"
for file in files(path):
    if "QueryResults_" in file:
        temp = pd.read_csv(os.path.join(path, file))
        df = pd.concat([df, temp])
        print(os.path.join(path, file), df.shape)

titles = df[['PostId','ParentId','CommentId','Title']].drop_duplicates(subset= ['PostId','Title']).dropna(subset = ['PostId','Title']).reset_index(drop = True).copy()
titles['category'] = 'title'
titles.rename(columns={'PostId': 'post_id','ParentId':'parent_id', 'CommentId': 'comment_id', 'Title':'text'}, inplace = True)
titles['comment_id'] = None
titles['parent_id'] = None

posts = df[['PostId','ParentId','CommentId', 'Body']].drop_duplicates(subset= ['PostId','ParentId','Body']).dropna(subset= ['PostId','Body']).reset_index(drop = True)
posts['category'] = 'post'
posts.rename(columns={'PostId': 'post_id','ParentId':'parent_id', 'CommentId': 'comment_id', 'Body':'text'}, inplace = True)
posts['comment_id'] = None

comments = df[['PostId','ParentId','CommentId', 'Text']].drop_duplicates(subset= ['PostId','CommentId','Text']).dropna(subset= ['PostId','CommentId', 'Text']).reset_index(drop = True)
comments['category'] = 'comment'
comments.rename(columns={'PostId': 'post_id','ParentId':'parent_id', 'CommentId': 'comment_id', 'Text':'text'}, inplace = True)
comments['parent_id'] = None


data = pd.concat([titles, posts,comments], sort=False)

data.to_csv('../data/stackexchange_812k.csv', quoting = csv.QUOTE_ALL, index = False)

# ------------------------------------------------------------------------------

'''
clean up with regex
'''
# rm html tags
data['text'] = data.text.apply(lambda t : re.sub("<[^>]*>",' ', t) )
# rm line returns
data['text'] = data.text.apply(lambda t : re.sub("[\r\n]+",' ', t) )
# rm urls
data['text'] = data.text.apply(lambda t : re.sub("http\S+",' ', t) )

# strip
data['text'] = data.text.apply(lambda t : t.strip() )



# ---------------------------------------------------------------------------
# Version 01
# ---------------------------------------------------------------------------

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
