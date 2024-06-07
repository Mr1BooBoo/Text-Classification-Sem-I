# -*- coding: utf-8 -*-
"""NLP_HW1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CVRnEfKJ88IOjWg4tdjjqyhx9aA3ERtb
"""

import nltk

nltk.download("book")

from nltk.book import *

text2 = Text(webtext.words('grail.txt'), name = "Sense and Sensibility by Jane Austen 1811")

type(text2)

len(text2)

text2

nltk.download('punkt')

"""# Tokenize the text"""

from nltk import sent_tokenize
import string
tokenized = [nltk.sent_tokenize(text) for text in text2]
len(tokenized)

"""# Remove punctuation"""

string.punctuation

og_no_punct = [word for word in tokenized if word[0] not in string.punctuation]
print(len(og_no_punct))


#The only one-letter words in English are a and I. therefore i eliminated words with 1 letter that are not "aA" or "iI"
no_punct = [word for word in og_no_punct if len(word[0]) > 1 or word[0] in ['a','A','i','I']]
print(no_punct[:10])
len(no_punct)

#from the following code we observe that the no_punct list contains lists of strings like this [['word1'],['word2'],['word3']]
print(type(no_punct))
print(type(no_punct[0]))

#so i wrote a function that will return a list of strings directly without being a list of lists of strings
def extractDigits(lst):
    return [el for [el] in lst]
no_punct = extractDigits(no_punct)
print(no_punct[:10])

pip install num2words

#to remove numbers, alternativly we could simply convert them
'''my_list = [item for item in tokenized if item.isalpha()]
len(my_list)'''
#to convert numbers:
from num2words import num2words
no_punct = [num2words(word) if word.isnumeric() else word for word in no_punct]
len(no_punct)

counter= 0
for word in no_punct[:]:
  
  if len(word) == 13:
    counter += 1
    print(word)
print('nr of words: ',counter)

def get_words_of_len(lg):
  counter = 0
  for word in no_punct:
    #print(len(word[0]))
    if len(word) == lg:
      counter += 1
  return  counter
get_words_of_len(5)

def get_starting_words(lt):
  good_words = []
  no_punct.sort()
  for word in no_punct:
    if word.startswith(lt):
      good_words.append(word)
  return good_words

words = get_starting_words('S')
#words

#Print the shortest and longest words (alphanumeric strings). If there are multiple words with minimum or maximul length, you must print all of them.
min = 10
max = 0
for word in no_punct:
  if len(word) > max:
    max = len(word)
  if len(word) < min:
    min = len(word)
print(min,max)

for word in no_punct:
  if len(word) == min or len(word) == max:
    print(word)

#Print the first N most frequent words (alphanumeric strings) together with their number of appearances.
from collections import Counter

def most_frequent(List,n):
    occurence_count = Counter(List)
    return occurence_count.most_common(n)

print(most_frequent(no_punct,3))

#Print the medium length of the words in the text.
def get_avg_word_len(List):
  total_avg = sum(map(len, List)) / len(List)
  return total_avg
get_avg_word_len(no_punct)

#Print the words that appear only once in the text.
from collections import Counter

def unique(List):
    occurence_count = Counter(List)
    unique = [key for key,val in occurence_count.items() if int(val) == 1]
    return unique 

print(unique(no_punct))

#Print the collocations in the text
#nltk.corpus.genesis.words('english-web.txt')
import nltk
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
fourgram_measures = nltk.collocations.QuadgramAssocMeasures()
finder = BigramCollocationFinder.from_words(no_punct)
finder.nbest(bigram_measures.pmi, 10)

"""# different approaches"""

import urllib.request

#Download it through python with urlopen() from module urllib and read the entire text in one single string.
target_url = 'https://www.gutenberg.org/cache/epub/70224/pg70224-images.html'
data = urllib.request.urlopen(target_url) # it's a file like object and works just like a file
raw_data = data.read()
'''type(data)
for line in data: # files are iterable
    print(line)'''
print(raw_data)

from urllib.request import urlopen

data = urlopen(target_url)
for line in data: # files are iterable
    print(line)

txt = urlopen(target_url).read()
txt

"""# working Ex2"""

from urllib import request
url = "https://www.gutenberg.org/files/70224/70224-0.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')
type(raw)
len(raw)

raw[:75]

import nltk
nltk.download('punkt')
from nltk import word_tokenize

#Remove the header (keep only the text starting from the title)
tokens = word_tokenize(raw)
#tokens[:9]
tokens.pop(0)
tokens[:9]

#Print the number of sentences in the text. Print the average length (number of words) of a sentence.
len(tokens)

avg_len = sum( map(len, tokens) ) / len(tokens)
avg_len

#Find the collocations in the text (bigram and trigram). Use the nltk.collocations module You will print them only once not each time they appear.
import nltk
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
fourgram_measures = nltk.collocations.QuadgramAssocMeasures()
finder = BigramCollocationFinder.from_words(tokens)
finder.nbest(bigram_measures.pmi, 10)

#Create a list of all the words (in lower case) from the text, without the punctuation.
from nltk import sent_tokenize
import string
string.punctuation

og_no_punct = [word for word in tokens if word not in string.punctuation]
print(len(og_no_punct))
no_punct = [word for word in og_no_punct if len(word) > 1 or word in ['a','A','i','I']]
print(no_punct[:10])
len(no_punct)

#Print the first N most frequent words (alphanumeric strings) together with their number of appearances.
from collections import Counter

def most_frequent(List,n):
    occurence_count = Counter(List)
    return occurence_count.most_common(n)

print(most_frequent(no_punct,3))

nltk.download('stopwords')

#Remove stopwords and assign the result to variable lws
from nltk.corpus import stopwords
from collections import Counter
stop_words = stopwords.words('english')
stopwords_dict = Counter(stop_words)
lws = [word for word in no_punct if word not in stopwords_dict]
print(lws[:10])

#Apply stemming (Porter) on the list of words (lws). Print the first 200 words. Do you see any words that don't appear in the dictionary?

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
  
ps = PorterStemmer()

for word in lws[:200]:
    print(word, " : ", ps.stem(word))
