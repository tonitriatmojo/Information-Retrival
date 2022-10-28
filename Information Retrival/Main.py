from Preprocessing import *
from Likelihood import *
from TFIDF import *
from BM25 import *
from TestResults import *
from scipy.stats import ttest_ind
import sys


if __name__ == "__main__":
    """
    getArticleDic() is passed the path for the Data Collection folder and stems the text and breaks down each new article to
    a dictioanry with the structure dic[folderName][Docid][term:frequency]. A dictionary containing the term frequency for
    every file is returned
    """
    articleDic = getArticleDic('DataCollection') #
    """
    getTitles() is passed the topics file path and get the title and folderID for all 50 folders and returns a dictionary of
    folderID and title(which is used as query)
    """
    dataTitle = getTitles("Topics.txt")

    """"
    this runs the likelihood model and stores all relevent values (values greate then 0)
    in a variable. Top 10 values are outputed for each query.
    as this is the first of three codes to run the model and all three are the same I will only comment this
    one.
    """
    """
    the likelihoodRetrived dictionary stores the sorted values returned from the likelihood models
    where the values are above 0.
    """
    LikelihoodRetrived = {}
    print("     ")
    print("Results for Query of Likelihood model")
    print("     ")
    """
    loop through data tiles to get the folder ID and the title which will be the query
    """
    for id, query in dataTitle:
        """
        this list is declared after the loop because it is only meant to hold the documentID of relevant docs
        it is later added to a dictionary with the key of the folder, making  a dictionary[folderName]:reldocs,
        this is used for testing.
        """
        LikelihoodRetrivedList = []
        print("Topic {}".format(id))
        print("Text used for query is: {}".format(query))
        """
        the id from the dataTitle is in "R101" format and needs to be converted to
        dataset101 format for the dictionary
        """
        coll = str("Dataset" + id[1:])
        """
        parse_query() is passed the query to format it into a dictionary of term:frequency
        """
        queryDic = parse_query(query)
        """
        IRLikelihood model is pased the parsed query and articleDic for a particular folder
        """
        likelihoodResult = IRLikelihood(articleDic[coll], queryDic)
        """
        results are sorted by the relevance score of the function
        """
        likelihoodSorted = sorted(likelihoodResult.items(), key=lambda x: x[1],reverse=True)
        print('DocID        Weight')
        """
        docID and value are looped through the resultSorted list and if value is above 0 the docID is added to the LikelihoodRetrivedList
        and then added to a dictionary with the key being the folder ID
        """
        for doc, value in likelihoodSorted:
            if value > 0.0000001:
                LikelihoodRetrivedList.append(doc)
        LikelihoodRetrived[id] = LikelihoodRetrivedList
        """"
        the top 10 results for resultsort are outputted granted they are greater then the threshold 1
        """
        for (doc, value) in likelihoodSorted[:10]:
            if value > 0.0000001:
                print("{}       {}".format(doc, str(value)))

    """"
    this runs the TFIDF model and stores all relevent values (values greate then 0)
    in a variable. Top 10 values are outputed for each query
    """
    TFIDFRetrived = {}
    print("     ")
    print("Results for Query of TFIDF model")
    print("     ")
    for id, query in dataTitle:
        TFIDFRetrivedDocs = []
        print("Topic {}".format(id))
        print("Text used for query is: {}".format(query))
        coll = str("Dataset" + id[1:])
        queryDic = parse_query(query)
        calwordfreq = calc_df(articleDic[coll])
        TFIDFScore = getTFIDFScore(articleDic[coll], calwordfreq, queryDic)
        TFIDFSorted = sorted(TFIDFScore.items(), key=lambda x: x[1],reverse=True)
        print('DocID        Weight')
        for doc, value in TFIDFSorted:
                if value > 0.01:
                    TFIDFRetrivedDocs.append(doc)
        TFIDFRetrived[id] = TFIDFRetrivedDocs
        for doc, value in TFIDFSorted[:10]:
            if value> 0.01:
                    print("{}       {}".format(doc, value))
        print("   ")

    """"
    this runs the BM25 model and stores all relevent values (values greate then 0)
    in a variable. Top 10 values are outputed for each query
    """
    BM25Retrived = {}
    print("     ")
    print("Results for Query of BM25 model")
    print("     ")

    for id, query in dataTitle:
        BM25RetrivedDocs = []
        print("Topic {}".format(id))
        print("Text used for query is: {}".format(query))
        coll = str("Dataset" + id[1:])
        queryDic = parse_query(query)
        calwordfreq = calc_df(articleDic[coll])
        score4BM225 = score_BM25(articleDic[coll], queryDic, calwordfreq)
        BM25Sorted = sorted(score4BM225.items(), key=lambda x: x[1],reverse=True)
        print('DocID        Weight')
        for doc, value in BM25Sorted:
            if value > 0:
                BM25RetrivedDocs.append(doc)
        BM25Retrived[id] = BM25RetrivedDocs
        for doc, value in BM25Sorted[:10]:
            if value > 0:
                print("{}       {}".format(doc, value))
        print("         ")


    """
    The following code is used to test for Recall, Percision and F1 scores for all three
    models and return a pandas dataframe with comparing all three results aross these vareiables.
    All documents above the threshold of 0 are previeved as relevant, and all relevant scores where used
    in calculating these metrics
    """
    """
    getDsetRelDocsTuple() is passed the RelevanceFeedback folder and returns a tuple of folderID and the documents
    that are relevant to the query search.
    """
    relevantID = getDsetRelDocsTuple('RelevanceFeedback')
    """
    Build intial dataframe for all three metrics for likelohood model
    """
    likelihoodScores = getEval(relevantID, LikelihoodRetrived)
    Likelihoodpd = pd.DataFrame.from_dict(likelihoodScores, orient='index', columns=['Recall (Likelihood)', 'Percision (Likelihood)', 'F1 (Likelihood)'])
    """
    Build intial dataframe for all three metrics for TFIDF model
    """
    TFIDFScores = getEval(relevantID, TFIDFRetrived)
    TFIDFpd = pd.DataFrame.from_dict(TFIDFScores, orient='index', columns=['Recall (TFIDF)', 'Percision (TFIDF)', 'F1 (TFIDF)'])
    """
    Build intial dataframe for all three metrics for BM25 model
    """
    BM25Scores = getEval(relevantID, BM25Retrived)
    BM25pd = pd.DataFrame.from_dict(BM25Scores, orient='index', columns=['Recall (BM25)', 'Percision (BM25)', 'F1 (BM25)'])
    """
    The following code creates a pandas dataframe from all three models comparing them accros the same testing metrics
    """
    F1pd = pd.concat([TFIDFpd["F1 (TFIDF)"], BM25pd["F1 (BM25)"], Likelihoodpd["F1 (Likelihood)"]], axis=1)
    Percisionpd = pd.concat([TFIDFpd["Percision (TFIDF)"], BM25pd["Percision (BM25)"], Likelihoodpd["Percision (Likelihood)"]], axis=1)
    Recallpd = pd.concat([TFIDFpd["Recall (TFIDF)"], BM25pd["Recall (BM25)"], Likelihoodpd["Recall (Likelihood)"]], axis=1)
    print(F1pd)
    print("     ")
    print(Percisionpd)
    print("     ")
    print(Recallpd)
    print("     ")

    """
    Following code is for average recall, percision and F1 statisitics of all three algorthims
    """

    print("Mean of Recall score for Likelihood model is {}, TFIDF is {}, BM25 is {}".format(round(Likelihoodpd["Recall (Likelihood)"].mean(), 3), round(TFIDFpd["Recall (TFIDF)"].mean(), 3), round(BM25pd["Recall (BM25)"].mean(), 3)))
    print(" ")
    print("Mean of Percision score for Likelihood model is {}, TFIDF is {}, BM25 is {}".format(round(Likelihoodpd["Percision (Likelihood)"].mean(), 3), round(TFIDFpd["Percision (TFIDF)"].mean(), 3), round(BM25pd["Percision (BM25)"].mean(), 3)))
    print(" ")
    print("Mean of F1 score for Likelihood model is {}, TFIDF is {}, BM25 is {}".format(round(Likelihoodpd["F1 (Likelihood)"].mean(), 3), round(TFIDFpd["F1 (TFIDF)"].mean(), 3), round(BM25pd["F1 (BM25)"].mean(), 3)))


    """
    The following code is used to get and comare the signifigant difference between F1 scores for all three models.
    the spicy.stats library was used for this.
    """

    LHTF_Ttest = ttest_ind(Likelihoodpd['F1 (Likelihood)'], TFIDFpd['F1 (TFIDF)'])
    print("T-test for signifigant difference between F1 scores of Likelihood model and TFIDF model with signifigance set to P<0.05")
    print("P value = {}".format(LHTF_Ttest[1]))
    print("     ")
    LHBM_Ttest = ttest_ind(Likelihoodpd['F1 (Likelihood)'], BM25pd['F1 (BM25)'])
    print("T-test for signifigant difference between F1 scores of Likelihood model and BM25 model with signifigance set to P<0.05")
    print("P value = {}".format(LHBM_Ttest[1]))
    print("     ")
    TFBM_Ttest = ttest_ind(TFIDFpd['F1 (TFIDF)'], BM25pd['F1 (BM25)'])
    print("T-test for signifigant difference between F1 scores of TFIDF model and BM25 model with signifigance set to P<0.05")
    print("P value = {}".format(TFBM_Ttest[1]))


    """
    run to get outputs from the alorithims
    """
    orig_stdout = sys.stdout
    f = open('ThreeModelsOutputs.txt', 'w')
    sys.stdout = f
    LikelihoodRetrived = {}
    print("     ")
    print("Results for Query of Likelihood model")
    print("     ")
    for id, query in dataTitle:
        LikelihoodRetrivedList = []
        print("Topic {}".format(id))
        print("Text used for query is: {}".format(query))
        coll = str("Dataset" + id[1:])
        queryDic = parse_query(query)
        likelihoodResult = IRLikelihood(articleDic[coll], queryDic)
        likelihoodSorted = sorted(likelihoodResult.items(), key=lambda x: x[1],reverse=True)
        print('DocID        Weight')
        for doc, value in likelihoodSorted:
            if value > 0.0000001:
                LikelihoodRetrivedList.append(doc)
        LikelihoodRetrived[id] = LikelihoodRetrivedList
        for (doc, value) in likelihoodSorted[:10]:
            if value>0.0000001:
                print("{}       {}".format(doc, str(value)))
    TFIDFRetrived = {}
    print("     ")
    print("Results for Query of TFIDF model")
    print("Text used for query is: {}".format(query))
    print("     ")
    for id, query in dataTitle:
        TFIDFRetrivedDocs = []
        print("Topic {}".format(id))
        print("Text used for query is: {}".format(query))
        coll = str("Dataset" + id[1:])
        queryDic = parse_query(query)
        calwordfreq = calc_df(articleDic[coll])
        TFIDFScore = getTFIDFScore(articleDic[coll], calwordfreq, queryDic)
        TFIDFSorted = sorted(TFIDFScore.items(), key=lambda x: x[1],reverse=True)
        print('DocID        Weight')
        for doc, value in TFIDFSorted:
                if value > 0.001:
                    TFIDFRetrivedDocs.append(doc)
        TFIDFRetrived[id] = TFIDFRetrivedDocs
        for doc, value in TFIDFSorted[:10]:
            if value>0.001:
                print("{}       {}".format(doc, value))
        print("   ")

    BM25Retrived = {}
    print("     ")
    print("Results for Query of BM25 model")
    print("Text used for query is: {}".format(query))
    print("     ")

    for id, query in dataTitle:
        BM25RetrivedDocs = []
        print("Topic {}".format(id))
        print("Text used for query is: {}".format(query))
        coll = str("Dataset" + id[1:])
        queryDic = parse_query(query)
        calwordfreq = calc_df(articleDic[coll])
        score4BM225 = score_BM25(articleDic[coll], queryDic, calwordfreq)
        BM25Sorted = sorted(score4BM225.items(), key=lambda x: x[1],reverse=True)
        print('DocID        Weight')
        for doc, value in BM25Sorted:
            if value > 0:
                BM25RetrivedDocs.append(doc)
        BM25Retrived[id] = BM25RetrivedDocs
        for doc, value in BM25Sorted[:10]:
            if value > 0:
                print("{}       {}".format(doc, value))
        print("         ")
    f.close()
