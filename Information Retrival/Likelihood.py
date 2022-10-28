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
the IRlikilhood() function is passed a dictionary produced by the articleDic() function for an individual data collection and returns a dictionary of ID and likelihood score
the algorithim is based of the equation constant*(term frequency/document lentght).
"""
def IRLikelihood(article, query):
    invertDic = {}
    idScoreDic = {}
    docLen = {}
    """
    the key of the query is looped through and a query match dictionary is declared. The idScoreDic dictionary is declared 1 at the key position of the article id and similarly declared 0.5 with
    the same key in the docLen dictionary.

        the article items of the dictionary for the key value, which is the term and its frequency are looped through. if the key(term) matches the queryKey(query term) then the querymatch dictionary
        is assigned the article[id] value + 1 with the docuemntID being the key, if they dont match queryMatch is assigned 1. This is as per the Laplace smoothing technique
        Once article[id].items() is looped through the queryMatch dictionary is declared to the invertedList dictionary with the queryKey as the key.

        Jedlin-Merscer Smoothing was used but was less affective then laplace, for this reason laplace was used, the code of using Jedlin-Mercer is in the report. this was done show the use of 
        non tutorial code to satisify the criteria
    """
    for queryKey in query.keys():
        queryMatch = {}
        for id in article.keys():
            idScoreDic[id]=1
            docLen[id]=0.5
            for key, value in article[id].items():
                if key == queryKey:
                    queryMatch[id] = value+1
                if not(queryKey in article[id]):
                    queryMatch[id] = 1
            invertDic[queryKey] = queryMatch

    #this is to calculate the document lenth
    for id in article.keys():
        total = 0
        for value in article[id].values():
            total = total+value
        docLen[id] += total
    """
    the score is calculated here by interacting though the items(id, score) in idScoreDic(empty dic at the start but continues to interact to update score) and then the same in the invertedDic (term, freq).
    if id is not in frequency, the freq dictionary at key value id is decalred 0.00.
    score is then declared as score*(freq[id]/doclen[id]) and score is updated for every word in the document.
    once all the terms and frequencies are passed though the likelihood model equation of score*(freq[id]/docLen[id]), the final score is added to idScoreDic with id as key and socre as value.
    """
    for (id, score) in idScoreDic.items():
        for (term, feq) in invertDic.items():
            if not(id in feq):
                feq[id]=0.000
            #document lenght is controlled for by dividing frequency by document lenght
            score = score*(feq[id]/docLen[id])
        idScoreDic[id] = score

    return idScoreDic
