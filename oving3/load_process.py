import random; random.seed(123)
import codecs
from nltk.stem.porter import PorterStemmer
import string
from gensim import corpora, models, similarities
import warnings
import pprint
warnings.simplefilter(action='ignore', category=FutureWarning)


stemmer = PorterStemmer()
stopWords = set("a,able,about,across,after,all,almost,also,am," \
            "among,an,and,any,are,as,at,be,because,been,but," \
            "by,can,cannot,could,dear,did,do,does,either,else," \
            "ever,every,for,from,get,got,had,has,have,he,her," \
            "hers,him,his,how,however,i,if,in,into,is,it,its," \
            "just,least,let,like,likely,may,me,might,most,must," \
            "my,neither,no,nor,not,of,off,often,on,only,or," \
            "other,our,own,rather,said,say,says,she,should,since," \
            "so,some,than,that,the,their,them,then,there,these,they," \
            "this,tis,to,too,twas,us,wants,was,we,were,what," \
            "when,where,which,while,who,whom,why,will,with,would," \
            "yet,you,your".split(","))

def remove_punctuation(sentence):
    punct = string.punctuation + "\r\n"
    sentence = sentence.translate({ord(c): None for c in punct})
    return sentence.lower()


def tokenise(sentence):
    return sentence.split()


def document_to_list():
    f = codecs.open("pg3300.txt", "r", "utf-8")
    paragraph = ""
    paragraphList=[]


    """boolean value used to detect several newline characters in a row, i.e end of a paragraph."""
    prevLineNewline = False
    """Loop over f.readlines(). If element is not a single newline character: add the element to the paragraph string.
    If element is a single newline character: remove punctuation and whitespace characters from paragraph strings.
    Add paragraph to paragraphList"""
    for element in f.readlines():
        if element == "\r\n":
            if not prevLineNewline: #if previous line was not a newline charachter
                prevLineNewline = True  #indicate that last readline was a newline character
                paragraphList.append(paragraph.strip())
                paragraph = ""
            else:
                pass
        else:
            paragraph += element + " "
            if prevLineNewline:
                prevLineNewline = False
    return paragraphList
    f.close()



"""remove all paragraphs with the word 'Gutenberg'"""
def exclude_Gutenberg(paragraphList):
    excludeGutenbergList = []
    for element in paragraphList:
        if "gutenberg" not in element.lower():
            excludeGutenbergList.append(element)
    return excludeGutenbergList



"""tokenise paragraphs, remove punctuation and newline chars, and convert to lower case"""
def tokenise_paragraphs(paragraphList):
    punct = string.punctuation + "\r\n\t"
    tokenisedParagraphList = []
    for element in paragraphList:
        remove_punctuation(element) # remove punctiations from paragraph
        tokenisedParagraphList.append(element.lower().split(" "))
    return tokenisedParagraphList

"""remove stopwords from list of tokenised paragraphs"""
def remove_stopwords(list_of_lists):
    list_of_lists_without_stopwords = []
    for paragraph in list_of_lists:
        list_of_lists_without_stopwords.append([])
        for word in paragraph:
            if word not in stopWords:
                list_of_lists_without_stopwords[-1].append(word)

    return list_of_lists_without_stopwords

def stem_1d_list(oneDList):
    stemmer = PorterStemmer()
    stemmedList = [stemmer.stem(token) for token in oneDList]
    return stemmedList


"""stem tokenised list"""
def stem_2d_list(tokenisedList):
    stemmer = PorterStemmer()
    stemmedList = []
    for paragraph in tokenisedList:
        stemmedList.append(stem_1d_list(paragraph))
    return stemmedList



"""building a dictionary.
Takes in list of tokenised paragraphs.
returns coropa.Dictionary type object of all words from of tokens argument"""
def build_dictionary(pocessedText):
    indexDict = corpora.Dictionary(processedText)
    return indexDict



"""filter out stop words from dictionary"""
def filter_stopwords_from_dictionary(indexDict):
    stopWordIDs = []
    for stopWord in stopWords:
        try:
            stopWordIDs.append(indexDict.token2id[stopWord])
        except KeyError:
            pass
    indexDict.filter_tokens(stopWordIDs)
    return indexDict


def convert_to_vec(dict, sentence):
    return dict.doc2bow(sentence)

"""create corpus from dictionary. returns """
def create_corpus(indexDict, processedText):
    corpus = [indexDict.doc2bow(paragraph) for paragraph in processedText]
    return corpus




"""make TF-IDF model based on corpus"""
def create_tfIdf_model(corpus):
    tfIdfModel = models.TfidfModel(corpus)
    return tfIdfModel


"""create TF-IDF weighted list """
def create_tfidf_list(corpus, tfIdfModel):
    weighted_list = []
    for paragraph in corpus:
        weighted_list.append(tfIdfModel[paragraph])
    return weighted_list



def create_lsi_model(tfIdfWeightedCorpus, indexDictionary, numTopics):
    return models.LsiModel(tfIdfWeightedCorpus, id2word=indexDictionary, num_topics=numTopics)



def create_latent_space(weightedCorpus,lsiModel):
    return lsiModel[tfIdfWeightedCorpus]



def create_TFIDS_similarity_matrix(tfIdfWeightedCorpus):
    return similarities.MatrixSimilarity(tfIdfWeightedCorpus)



def create_LSI_sim_matrix(lsiCorpus):
    return similarities.MatrixSimilarity(lsiCorpus)



def process_query(query):
    query = remove_punctuation(query)
    query = tokenise(query)
    query = stem_1d_list(query)
    query
    return query



#Load document and represent document as a list of paragraphs
listOfParagraphs = document_to_list()

#remove paragraphs with the word gutenberg
originalParagraphs = exclude_Gutenberg(listOfParagraphs)

#tokenise paragraphs. returns a list containing paragraphs represented as lists of tokens
tokenisedParagraphs = tokenise_paragraphs(originalParagraphs)


tokenisedParagraphsWithoutStopwords = remove_stopwords(tokenisedParagraphs)


#stem words in document
processedText = stem_2d_list(tokenisedParagraphsWithoutStopwords)

#creates a dictionary of all words in the document and their frequency
indexDictionary = build_dictionary(processedText)


#create a corpus based on the entire document
corpus = create_corpus(indexDictionary, processedText)

#create TF-IDF model based on corpus
tfIdfModel = create_tfIdf_model(corpus)

#create TF-IDF weighted
tfIdfWeightedCorpus = create_tfidf_list(corpus,tfIdfModel)

#create LSI model based on the corpus with TF-IDF weightes and indexDictionary with 100 topics
lsiModel = create_lsi_model(tfIdfWeightedCorpus, indexDictionary, numTopics=100)

#create latent 100d space based on TF-IDF weighted corpus:
lsiCorpus = create_latent_space(tfIdfWeightedCorpus, lsiModel)

#Create similarity matrix for TF-IDF weighted documents and queries:
tfIdf_index = create_TFIDS_similarity_matrix(tfIdfWeightedCorpus)

#create similarity matrix for LSI:
lsi_index = create_LSI_sim_matrix(lsiCorpus)

#process query. remove punctuation, convert to lower case, tokenise, stem
query = process_query("How taxes influence Economics?")

#convert query to sparse vector
query_vec = convert_to_vec(indexDictionary, query)

#Weight query with TF-IDF weights
tfidf_query = tfIdfModel[query_vec]

tfIdf_sims = sorted(list(enumerate(tfIdf_index[tfidf_query])),
                    key=lambda t: t[1], reverse=True)[:3]

lsi_query = lsiModel[tfidf_query]

top_query_topics = sorted(lsi_query, key=lambda t: -abs(t[1]))[:3]
for topic in top_query_topics:
    print("topic " + str(topic[0]))
    print(lsiModel.show_topic(topic[0]))
    print('\n')
print(sorted(enumerate(lsi_index[lsi_query]), key=lambda t: -t[1])[:3])
#print(tfIdf_sims)

def print_3_most_relevant_tfidf(tfIdfSimilarityList, originalParagraphsList):
    for element in tfIdfSimilarityList:
        print("paragraph " + str(element[0]) + ":")
        t = 0
        paragraph = originalParagraphsList[element[0]].split('\n')
        while paragraph and t < 5:
            print(paragraph.pop(0).strip())
            t += 1
        print('\n')
#print_3_most_relevant_tfidf(tfIdf_sims, originalParagraphs)


#for i in range (3):
 #   t = sims[i][0]
  #  for
   # listOfParagraphsExcludeGutenberg[t].split('\n')

"""
print(query_tfIdf_vec)

print(indexDictionary.id2token[2739] + "\t" + indexDictionary.id2token[4036] + "\t" + indexDictionary[6277])
"""
"""
sims = tfIdsSimMatrix[tfIdf_query_bow]
sorted = sorted(list(enumerate(sims)), key=lambda t: t[1], reverse=True)
print(sorted)
for i in range(3):
    paragraphNum = sorted[i][0]
    paragraph = paragraphList[paragraphNum]
    print("Paragraph " + str(paragraphNum) + paragraph)
    print("\n\n")
"""