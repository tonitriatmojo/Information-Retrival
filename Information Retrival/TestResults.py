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
the following code is used to go through the RelevanceFeedback document and return a list of the folderID and all the documents that are relevant
the format is [folderID, [DocuemntID, documentID, documentID]]
"""

#getting relevant doc IDs for queires
def getFeedback(path): #put all feedbacks into an array by each line
    txts = os.listdir(path)
    wholefeedback=[]
    for txt in txts:
        filePath = os.path.join(path, txt)
        feedback = open(filePath,'r').read() #this had to be changed for difference in operating system, may have to edit for yours
        lines = feedback.split('\n')
        lines.remove('')
        for line in lines:
            wholefeedback.append(line)
    return wholefeedback

def getDsetRelDocs(path): #Remove all 0 relevence feedback
    relFeedback = getFeedback(path)
    removelist = []
    for feedback in relFeedback:
        if feedback.endswith('0'):
            removelist.append(feedback)
    while len(removelist) != 0 :
        for feedback in relFeedback:
            if feedback in removelist:
                relFeedback.remove(feedback)
                removelist.remove(feedback)
    return relFeedback

def getDsetRelDocsTuple(path): #Ruturn tuple of {dataset : [relevence doc(s) id]}
    dataSetRelDocsTuple = {}
    orderTag = []
    orderDic = {}
    for feedback in getDsetRelDocs(path):
        ids = feedback.split(' ')
        tupleBuffer = ids[1]
        try:
            dataSetRelDocsTuple[ids[0]].append(tupleBuffer)
        except (KeyError):
            dataSetRelDocsTuple[ids[0]] = [tupleBuffer]
    for key in dataSetRelDocsTuple.keys():
        orderTag.append(key[1:])
    orderTag.sort()
    for tag in orderTag:
        orderDic["R{}".format(tag)] = dataSetRelDocsTuple["R{}".format(tag)]

    return orderDic

def findRelevantDocs(relDocs, retrivedDoc):
    relDocsDic = {}
    for key, relList in relDocs.items():
        totalRelDocs = 0
        for value in retrivedDoc[key]:
            if value in relList:
                totalRelDocs = totalRelDocs +1
        relDocsDic[key] = totalRelDocs
    return relDocsDic

def getEval(RelDocs, RetrievedDocs):
    #recallDic = {}
    #percisionDic = {}
    #F1Dic = {}
    collDic = {}
    relTFIDF = findRelevantDocs(RelDocs, RetrievedDocs)
    for key in RelDocs.keys():
        evalList = []
        if relTFIDF[key] + len(RelDocs[key]) == 0:
            recall = 0
        else:
            recall = float(relTFIDF[key])/float(len(RelDocs[key]))
        if relTFIDF[key] + len(RetrievedDocs[key]) == 0:
            percision = 0
        else:
            percision =  float(relTFIDF[key])/float(len(RetrievedDocs[key]))

        if percision+recall == 0:
            F1score = 0
        else:
            F1score = float(2*percision*recall)/float((percision+recall))

        evalList.append([recall, percision, F1score])
        collDic[key] = [recall, percision, F1score]
    return collDic
