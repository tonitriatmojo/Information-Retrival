import pandas as pd
import nltk as nl
from nltk.stem.porter import *
from nltk.corpus import stopwords
import sklearn as sk
import numpy as np
import matplotlib as mp
import os
import string
import math


"""
The calc_df() function is passed the article dictionary with the key for the folder of articles that is being referenced.
an empty df_={} is declared and the keys of the dictionary position that is passed is looped thoughh. These keys are used as the
keys for then next loop of the artDIc to get its next keys which are the terms of the document. each term is used as the key for the the
df_ dictioanry to add 1 for each document it occurs in or to decalre 1 if its its first occurnance
"""
def calc_df(artdic):
    df_ = {}
    for id in artdic.keys():
        for key in artdic[id].keys():
            try:
                df_[key] += 1
            except KeyError:
                df_[key] = 1
    return df_

"""
getTFIDF() is a function to get a TFIDF score for the document of the dictionary collection that is passed (function is passed articleDic[folder][docID]).
tf is calculated by dividing the terms frequency by the total words in the documents, this value is then decalred to tdDict with the term as key and tf score as value.
idf is calculated by getting the natural log of the number of documents in the collection divdided by the terms document frequency.
tfidf is calculated by timesing the tf and idf of a term together. this returns a dictioanry of terms and their TFIDF scores for that documents.

"""
def getTFIDF(artDic, df):
    tfDict = {}
    docFreq = df
    docTotal = 0
    numDoc = len(artDic)

    for value in artDic.values():
        docTotal = docTotal+value
    for key, value in artDic.items():
        tfDict[key] = value/float(docTotal)

    idfDict = {}
    for word, val in docFreq.items():
        idfDict[word] = math.log10(numDoc / float(val))

    tfidfDic = {}
    for word, val in tfDict.items():
        tfidfDic[word] = val * idfDict[word]
    return tfidfDic

"""
this function uses the query terms and the the calculated tfidif of a documents terms the caclulate the tfidf of a document relevant to the query.
the tfidf of the documents term and the frequency of the query terms occurance are multiplied and appeded to a queryValue list, the sum of that list is decalred to a TFIDFScore dictionary with the DocID
as the key and the sum of query TDFID scores for that doc appended as the value.
"""
def getTFIDFScore(artDic, df, qDic):
    TFIDFScore = {}

    for doc in artDic.keys():
        exOutput = getTFIDF(artDic[doc], df)
        queryValue = []
        for key, value in qDic.items():
            try:
                queryValue.append(exOutput[key]*value)
            except KeyError:
                pass
        TFIDFScore[doc] = sum(queryValue)
    return TFIDFScore
