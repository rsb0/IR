import random; random.seed(123)
import codecs
from nltk.stem.porter import PorterStemmer
import string
from gensim import corpora, models, similarities
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

stemmer = PorterStemmer()
stopWords = "a,able,about,across,after,all,almost,also,am," \
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
            "yet,you,your".split(",")

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
paragraphList = document_to_list()[100]


"""remove all paragraphs with the word 'Gutenberg'"""
def exclude_Gutenberg(paragraphList):
    excludeGutenbergList = []
    for element in paragraphList:
        if "gutenberg" not in element.lower():
            excludeGutenbergList.append(element)
    return excludeGutenbergList
paragraphList1 = exclude_Gutenberg(paragraphList)


"""tokenise paragraphs, remove punctuation and newline chars, and convert to lower case"""
def tokenise_paragraphs(paragraphList):
    punct = string.punctuation + "\r\n\t"
    tokenisedParagraphList = []
    for element in paragraphList:
        remove_punctuation(element) # remove punctiations from paragraph
        tokenisedParagraphList.append(element.lower().split(" "))
    return tokenisedParagraphList
tokenised_paragraphs = tokenise_paragraphs(paragraphList1)

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
processedText = stem_2d_list(tokenised_paragraphs)


"""building a dictionary.
Takes in list of tokenised paragraphs.
returns coropa.Dictionary type object of all words from of tokens argument"""
def build_dictionary(pocessedText):
    indexDict = corpora.Dictionary(processedText)
    return indexDict
indexDictionary = build_dictionary(processedText)


"""filter out stop words from dictionary"""
def filter_stopwords_from_dictionary(indexDict):
    stopWordIDs = []
    for stopWord in stopWords:
        try:
            stopWordIDs.append(indexDict.token2id[stopWord])
            print(stopWord + "\t finnes i dict")
        except KeyError:
            print(stopWord + "\t finnes ikke i dict")
    indexDict.filter_tokens(stopWordIDs)
    return indexDict
indexDict = filter_stopwords_from_dictionary(indexDictionary)

def convert_to_bow(dict, sentence):
    return dict.doc2bow(sentence)

"""create corpus from dictionary. returns """
def create_corpus(indexDict, processedText):
    corpus = [convert_to_bow(indexDict, paragraph) for paragraph in processedText]
    return corpus


corpus = create_corpus(indexDict, processedText)
print(corpus)[100]


"""make TF-IDF model based on corpus"""
def create_tfIdf_model(corpus):
    tfIdfModel = models.TfidfModel(corpus)
    return tfIdfModel
TFIDFModel = create_tfIdf_model(corpus)

"""create TF-IDF weighted list """
def create_tfidf_list(corpus, tfIdfModel):
    weighted_list = []
    for paragraph in corpus:
        weighted_list.append(tfIdfModel[paragraph])
    return weighted_list
tfIdfWeightedCorpus = create_tfidf_list(corpus,TFIDFModel)


def create_lsi_model(tfIdfWeightedCorpus, indexDictionary, numTopics):
    return models.LsiModel(tfIdfWeightedCorpus, id2word=indexDictionary, num_topics=numTopics)
lsiModel = create_lsi_model(tfIdfWeightedCorpus, indexDictionary, 100)


def create_lsi_list(weightedCorpus,lsiModel):
    return lsiModel[tfIdfWeightedCorpus]
lsiCorpus = create_lsi_list(tfIdfWeightedCorpus, lsiModel)


def create_TFIDS_sim_matrix(tfIdfWeightedCorpus):
    return similarities.MatrixSimilarity(tfIdfWeightedCorpus)
tfIdsSimMatrix = create_TFIDS_sim_matrix(tfIdfWeightedCorpus)


def create_LSI_sim_matrix(lsiCorpus):
    return similarities.MatrixSimilarity(lsiCorpus)
lsiSimMatrix = create_LSI_sim_matrix(lsiCorpus)


def process_query(query):
    query = remove_punctuation(query)
    query = tokenise(query)
    query = stem_1d_list(query)
    query
    return query
query = process_query("what is the function of money?")
print(indexDict.token2id["function"])
print(query)
query = convert_to_bow(indexDict, query)
print(query)

