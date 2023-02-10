import requests
import json
import os
from pathlib import Path
import pandas as pd
import re
from typing import Set
from transformers import GPT2TokenizerFast

import numpy as np
from nltk.tokenize import sent_tokenize

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    """count the number of tokens in a string"""
    return len(tokenizer.encode(text))

def reduce_long(
    long_text: str, long_text_tokens: bool = False, max_len: int = 590
) -> str:
    """
    Reduce a long text to a maximum of `max_len` tokens by potentially cutting at a sentence end
    """
    if not long_text_tokens:
        long_text_tokens = count_tokens(long_text)
    if long_text_tokens > max_len:
        sentences = sent_tokenize(long_text.replace("\n", " "))
        ntokens = 0
        for i, sentence in enumerate(sentences):
            ntokens += 1 + count_tokens(sentence)
            if ntokens > max_len:
                return ". ".join(sentences[:i][:-1]) + "."

    return long_text


url = "https://query.semlab.io/proxy/wdqs/bigdata/namespace/wdq/sparql"

sparql = """
	select * { 
	  {select * where { 
	    ?block wdt:P11 wd:Q19104.
	  }}
	  
	  ?block wdt:P24 ?parentDoc .
	  ?parentDoc rdfs:label ?parentDocLabel. 
	  ?block wdt:P20 ?textUrl .

	  
	}
"""

params = {
	'query' : sparql
}

headers = {
	'Accept' : 'application/json',
	'User-Agent': 'USER thisismattmiller - Test Script '
}

r = requests.get(url, params=params, headers=headers)

data = json.loads(r.text)

docs = []

for result in data['results']['bindings']:


	qid = result['block']['value'].split('/')[-1]
	url = result['textUrl']['value']
	doc = result['parentDocLabel']['value']


	if os.path.exists(f'block_text/{qid}.txt'):


		text = Path(f'block_text/{qid}.txt').read_text()

		if 'Martin:' in text:
			text = text.replace('Martin: ', 'An interview with Julie Martin: ')
			text=text.strip()


		if 'Story of E.A.T.' in doc:
			text = 'The Story of E.A.T. by Billy Kluver: ' + text

			text=text.strip()

		if count_tokens(text) > 1000:

			
			text = text.split('\n')
		else:
			text = [text]


		for t in text:

			trup = (doc,qid,t,count_tokens(t))

			docs.append(trup)





df = pd.DataFrame(docs, columns=["title", "heading", "content", "tokens"])
df = df[df.tokens>40]
df = df.drop_duplicates(['title','heading'])
df = df.reset_index().drop('index',axis=1) # reset index
df.head()

df.to_csv('docs.csv', index=False)

print(df.title.value_counts().head())
	