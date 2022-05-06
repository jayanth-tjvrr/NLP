import nltk
import nltk.collocations
import nltk.corpus
import collections
webtext =nltk.corpus.webtext 
bigrams = nltk.collocations.BigramAssocMeasures()

words = [w.lower() for w in webtext.words('4.txt')] 
bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(words)
bigramratio = bigramFinder.score_ngrams( bigrams.likelihood_ratio  )
bigramratio[:9]
