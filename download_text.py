import requests
import json
from html.parser import HTMLParser

from bs4 import BeautifulSoup

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

	print(qid,url,doc)
	headers = {
		'charset' : 'utf-8',
		'Content-Type': 'text/plain; charset=utf-8'
	}
	r = requests.get(url, headers=headers)
	if r.status_code == 200:

		r.encoding = 'utf-8'

		html_decoded_string = BeautifulSoup(r.text, "lxml");

		with open(f'block_text/{qid}.txt','w') as outfile:
			outfile.write(html_decoded_string.text)

	else:
		print(url,r.status_code)







	