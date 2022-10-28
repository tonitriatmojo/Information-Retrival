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
This function is passed the folder directory holding sub flolders which hold articles that are to be preprossessed by removing step words and counting word occurnance inside the article.
the function returns a dictionary in the structur of {"Dataset" : { docID : {word : frequency in int}}}
"""
def getArticleDic(path):
    articleDic = {}
    #this was to get rid of ".DS_store" files they kept coming up as file not being found when i didnt do this, may be a mac thing
    ds_store_file_location = path+'/.DS_store'
    if os.path.isfile(ds_store_file_location):
        os.remove(ds_store_file_location)
    for folder in os.listdir(path):
        folderPath = os.path.join(path, folder)
        folDic = {}
        for fileName in os.listdir(folderPath):
            filePath = os.path.join(folderPath, fileName)
            myfile = open(filePath)
            docid = fileName[:-4]
            file = myfile.readlines()
            docDic = {}
            for line in file:
                line = line.strip()
                if line.startswith("<p>"):
                    line = line.replace("<p>", "").replace("</p>", "")
                    line = line.translate(str.maketrans('','', string.digits)).translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
                    line = line.replace("\\s+", " ")
                    for term in line.split():
                        stemmer = PorterStemmer()
                        term = stemmer.stem(term.lower())
                        stop_words = set(stopwords.words('english'))
                        if len(term) > 2 and term not in stop_words:
                            try:
                                docDic[term] += 1
                            except KeyError:
                                docDic[term] = 1
                folDic[docid] = docDic
            articleDic[folder] = folDic
    return articleDic

"""
getTitles() is passed the file path and for the topixs document returns a dictionary of the corpus title "R101" and the title of the document in a list format.
getTitles() returns [docID, title]
"""
def getTitles(path):
    file = open(path)
    IDTitle = []
    for line in file:
        line = line.strip()
        if line.startswith("<num> "):
            for part in line.split():
                if part.startswith("R"):
                    docID = part
        if line.startswith("<title>"):
                #or part in line.split():
                docTitle = line[7:].strip(" ")
                IDTitle.append([docID, docTitle])
    return IDTitle
""""
Parse query is pased a the text of the title from get titles and reporduces a dictionary of the indiviual words in the title and thier frequency.
e.g. title = "Euro: European finance in trouble" {euro:2, financ:1, trouble:1}
"""
def parse_query(query):
    curr_doc = {}
    for line in query.replace('-', ' ').split(' '):
        line = line.strip()
        line = line.translate(str.maketrans('','', string.digits)).translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
        #print(line)
        #print('-----------')
        line = line.replace("\\s+", " ")
        #term = stem(term.lower()) ## for wk 4
        line = line.replace(" ", "")
        stemmer = PorterStemmer()
        line = stemmer.stem(line.lower()) #wk3
        stop_words = set(stopwords.words('english'))
        if len(line) > 2 and line not in stop_words: #wk3
            try:
                curr_doc[line] += 1
            except KeyError:
                curr_doc[line] = 1
    return curr_doc
