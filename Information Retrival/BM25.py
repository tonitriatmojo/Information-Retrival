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
avgDocLen() is passed the dictionary from the getArticleDic() function at the position
of one document collection, for example artDic["Dataset101"]. From here the function
itterates through all the documents and add the value (frequency) of each word to
the totalDocWords variable. Once all docuement words have been counted, the totalDocWords
is diveded by the number of document (len(artDic)) and the value is returned.
"""
def avgDocLen(artDic):
    numDoc = len(artDic)
    totalDocWords = 0
    for id in artDic.keys():
        docTotal = 0
        for value in artDic[id].values():
            docTotal = docTotal+value
        totalDocWords = totalDocWords+docTotal
    avgDocLen = totalDocWords/numDoc
    return avgDocLen
"""
calc_df() is passed the returned dictionary from the getArticleDic() at the position of one
collection (e.g. artDic[Dataset101]). The id of the artDic is iterrated through (artDic[Dataset101][6145])
and then passed as the key in the next for loop to itterate through the terms in that document. The loop will
first try to add 1 to the declared value for df_ for the key term but if it does not exist it then will
declare it as the first value. this is repeated for all documents in the collection.
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
This function passes the dictionary from the getArticleDic() with its first and second key, so artDic["collection"][id].
it then loops throught the values of the term:frequency dicinary adding all the values to the toralWords variable.
the totalWord variable is returned as just an interger

"""

#should pass the dictionary, so artdic[key][key] passed
def articleLen(artDIc):
    totalWords = 0
    for value in artDIc.values():
        totalWords = totalWords + value
    return totalWords

"""the score_BM25() function is passed a dictionary with its key value (artDic["Dataset"]), a query in string form (q), and document frequencies
of the terms (df). the score_BM25() uses the BM25 ranking algorithim to rank the documents on the likely relation
they have to the query that is passed through it. The function returns print statements  demonstrating the query
and the BM25 score, with the higher score meaning it is more likely that the document is relevant to that query.

The function startes by creating a stop word list and declaring an empty query result list (query_result).
Varaibles like K1, K2, b, ect. where set to either their predetermined values or in the case of K2 set to K2=100 (range 0-100).

Assumption: the output is not the same as the assignment sheet but the answers seem to scale to the same as the
example solutions so it my be a factor of log chosen or the setting of parameters like K2.

qf (query frequency) is calculated by using the parse_query function on the query pased though and similarly average
doc lenght (avdl) is calculated passing dictionary (artDic) through avg_doc_len().

a loop is then made to loop through each document in the artDic followed by a loop to for terms in query
frequency. n (number of ducements the term is in) is calculated by using the term as the key in the df dictionary, if
the word is present then it will define n as the value if not then n=0. Similarly this is done with f (terms frequency in doc)
using term as the key for the dictionary in the artDic.

K is calcualted using the compute_K() which gets passed document lenght (articleLen(artDic[id])) and average document lenght (avdl).
Commenting for compute_k() can be found above the functions' decleration.

All the varaibales decalred are then used to calculate the the BM25 score using the equation provided. the three
chunks of multiplecation are calculated serperatly and declared to the values first, second and third.
these thre vareiables are then all multiplied and declared as the value to the query_result dictionary
with the document id (doc[0]) set as the dictioanry key, this is then repeated for all docs.


"""

def score_BM25(artDic, q, df):
    #stopwords_f = open('common-english-words.txt', 'r')
    #stop_words = stopwords_f.read().split(',')
    #stopwords_f.close()
    query_result = dict()
    k1 = 1.2
    k2 = 500
    R = 0.0
    N = len(artDic)
    r = 0.0

    #qf = parse_query(q, stop_words)
    avdl = avgDocLen(artDic)
    for id in artDic.keys():
        for term in q.keys():
            try:
                n = df[term]
            except KeyError:
                n= 0
            try:
                f = artDic[id][term]
            except KeyError:
                f = 0
            K = compute_K(articleLen(artDic[id]), avdl)
            first = math.log10( ( (r + 0.5) / (R - r + 0.5) ) / ( (n - r + 0.5) / (N - n - R + r + 0.5)) )
            second = ((k1 + 1) * f) / (K + f)
            third = ((k2+1) * q[term]) / (k2 + q[term])
            score = first * second * third

            if id in query_result: #this document has already been scored once
                query_result[id] += score
            else:
                query_result[id] = score
    return query_result

"""
Calculation for k, b is set to 1 to maximise document normalisation and the document lenght is devided by the average document lenght to allow for normalisation of the algorithim
"""
def compute_K(dl, avdl):
	return 1.2 * ((1-1) + 1 * (float(dl)/float(avdl)) )
