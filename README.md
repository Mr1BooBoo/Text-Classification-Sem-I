# Text-Classification-Sem-I

In my first semester, I completed a text classification project aimed at processing and analyzing textual data using various Natural Language Processing (NLP) techniques. The project involved several key steps, leveraging libraries such as NLTK and TensorFlow.

#Project Steps:  
  
##Text Tokenization:

Utilized NLTK to tokenize text from "Sense and Sensibility" by Jane Austen.
Tokenized the text into sentences and words, removing punctuation and filtering out unnecessary characters.
Data Cleaning and Preprocessing:

Removed punctuation and filtered out one-letter words except 'a' and 'I'.
Converted numeric values to their word equivalents using the num2words library.
Identified and removed stopwords using NLTK's built-in list, and applied stemming using the PorterStemmer.
Word Analysis:

Implemented functions to analyze word lengths, starting letters, and frequency of words.
Identified the shortest and longest words, calculated average word lengths, and listed words that appear only once.
Collocations:

Extracted and identified collocations (common word pairs and triples) using NLTK's collocation measures.
Text from URL:

Downloaded and processed text directly from a URL, tokenizing and analyzing it similarly to the local text file.
Text Classification with Neural Networks:

Developed a text classification model using TensorFlow and Keras.
Preprocessed text data, including tokenization, lemmatization, and bag-of-words representation.
Built and trained a Convolutional Neural Network (CNN) to classify text into predefined categories based on training data.
Saved the trained model for future use in text classification tasks.
Tools and Technologies Used:
Programming Languages: Python
Libraries: NLTK, NumPy, TensorFlow, Keras, num2words
Techniques: Tokenization, Stemming, Lemmatization, Bag-of-Words, Convolutional Neural Networks
This project enhanced my understanding of NLP techniques and their practical applications in text processing and classification. It provided valuable hands-on experience with data cleaning, feature extraction, and building machine learning models for text data analysis.
