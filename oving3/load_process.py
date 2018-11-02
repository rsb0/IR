import random; random.seed(123)
import codecs
from nltk.stem.porter import PorterStemmer
import string
from gensim import corpora, models, similarities
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



stemmer = PorterStemmer()
stop_words = set("a,able,about,across,after,all,almost,also,am," \
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










"""-----------------------1: DATALOADING AND PREPOCESSING-----------------------"""

def remove_punctuation(sentence):
    punct = string.punctuation + "\r\n"
    sentence = sentence.translate({ord(c): None for c in punct})
    return sentence.lower()


def tokenise(sentence):
    return sentence.split()


def document_to_list():
    f = codecs.open("pg3300.txt", "r", "utf-8")
    paragraph = ""
    paragraph_list=[]

    """boolean value used to detect several newline characters in a row, i.e end of a paragraph."""
    prev_line_newline = False
    """Loop over f.readlines(). If element is not a single newline character: add the element to the paragraph string.
    If element is a newline character: remove punctuation and whitespace characters from paragraph strings.
    if two newline characters are encountered: Add paragraph to paragraphList"""
    for element in f.readlines():
        if element == "\r\n":
            if not prev_line_newline: #if previous line was not a newline charachter
                prev_line_newline = True  #indicate that last readline was a newline character
                paragraph_list.append(paragraph.strip())
                paragraph = ""
            else:
                pass
        else:
            paragraph += element + " "
            if prev_line_newline:
                prev_line_newline = False
    return paragraph_list
    f.close()



"""remove all paragraphs with the word 'Gutenberg'"""
def exclude_Gutenberg(paragraph_list):
    exclude_gutenberg_list = []
    for element in paragraph_list:
        if "gutenberg" not in element.lower():
            exclude_gutenberg_list.append(element)
    return exclude_gutenberg_list



"""tokenise paragraphs, remove punctuation and newline chars, and convert to lower case"""
def tokenise_paragraphs(paragraph_list):
    punct = string.punctuation + "\r\n\t"
    tokenised_paragraph_list = []
    for element in paragraph_list:
        remove_punctuation(element) # remove punctiations from paragraph
        tokenised_paragraph_list.append(element.lower().split(" "))
    return tokenised_paragraph_list



"""remove stopwords from list of tokenised paragraphs"""
def remove_stopwords(list_of_lists):
    list_of_lists_without_stopwords = []
    for paragraph in list_of_lists:
        list_of_lists_without_stopwords.append([])
        for word in paragraph:
            if word not in stop_words:
                list_of_lists_without_stopwords[-1].append(word)

    return list_of_lists_without_stopwords



"""stem 1d list of tokens"""
def stem_1d_list(oneDList):
    stemmer = PorterStemmer()
    stemmedList = [stemmer.stem(token) for token in oneDList]
    return stemmedList



"""stem tokenised list (list containing lists of tokens)"""
def stem_2d_list(tokenised_list):
    stemmer = PorterStemmer()
    stemmed_list = []
    for paragraph in tokenised_list:
        stemmed_list.append(stem_1d_list(paragraph))
    return stemmed_list









"""---------------------------------2: DICTIONARY BUILDING------------------------------------"""


"""building a dictionary.
Takes in list of tokenised paragraphs.
returns coropa.Dictionary object of document"""
def build_dictionary(pocessed_text):
    return corpora.Dictionary(processedText)



"""filter out stop words from dictionary"""
"""not in use"""
def filter_stopwords_from_dictionary(index_dict):
    stop_word_IDs = []
    for stop_word in stop_words:
        try:
            stop_word_IDs.append(index_dict.token2id[stop_word])
        except KeyError:
            pass
    index_dict.filter_tokens(stop_word_IDs)
    return index_dict


"""used to convert queries to vector of (word-index, word-frequency) tuples"""
def convert_to_vec(dict, sentence):
    return dict.doc2bow(sentence)


"""create corpus from dictionary"""
def create_corpus(index_dict, processed_text):
    corpus = [index_dict.doc2bow(paragraph) for paragraph in processed_text]
    return corpus








"""-------------------------------3: RETRIEVAL MODE-------------------------------"""


"""make TF-IDF model based on corpus"""
def create_tfIdf_model(corpus):
    tfidf_model = models.TfidfModel(corpus)
    return tfidf_model


"""apply TF-IDF weights to corupus"""
def apply_tfidf_model(model, corpus):
    return model[corpus]


"""make LSI model based on corpus"""
def create_lsi_model(tfidf_weighted_corpus, index_dictionary, numTopics):
    return models.LsiModel(tfidf_weighted_corpus, id2word=index_dictionary, num_topics=numTopics)


"""convert TF-IDF weighted corpus into LSI weighted"""
def create_latent_space(tfidf_weighted_corpus,lsi_model):
    return lsiModel[tfidf_weighted_corpus]


"""make TF-IDF similarity matrix based on corpus"""
def create_TFIDS_similarity_matrix(tfIdf_weighted_corpus):
    return similarities.MatrixSimilarity(tfIdf_weighted_corpus)


"""make LSI similarity matrix based on corpus"""
def create_LSI_sim_matrix(lsi_corpus):
    return similarities.MatrixSimilarity(lsi_corpus)



def process_query(query):
    query = remove_punctuation(query)
    query = tokenise(query)
    query = stem_1d_list(query)
    query
    return query

def print_3_most_relevant_tfidf(tfidf_similarity_list, original_paragraphs_list):
    for element in tfidf_similarity_list:
        print("paragraph " + str(element[0]) + ":")
        t = 0
        paragraph = original_paragraphs_list[element[0]].split('\n')
        while paragraph and t < 5:
            print(paragraph.pop(0).strip())
            t += 1
        print('\n')





"""----------------------Executing code-------------------------"""


"""Load document and represent document as a list of paragraphs"""
listOfParagraphs = document_to_list()


"""remove paragraphs with the word gutenberg"""
originalParagraphs = exclude_Gutenberg(listOfParagraphs)


"""tokenise paragraphs. returns a list containing paragraphs represented as lists of tokens"""
tokenisedParagraphs = tokenise_paragraphs(originalParagraphs)


"""Remove stopwords from original text before stemming and constructing dictionary. Commented out"""
#tokenisedParagraphsWithoutStopwords = remove_stopwords(tokenisedParagraphs)


"""stem words in document. Commented out"""
#processedText = stem_2d_list(tokenisedParagraphsWithoutStopwords)

"""stem list and make dictionary before removing stopwords:"""
processedText = stem_2d_list(tokenisedParagraphs)

"""creates a dictionary of all words in the document and their frequency"""
indexDictionary = build_dictionary(processedText)

"""filter stopwords from dictionary"""
indexDictionary = filter_stopwords_from_dictionary(indexDictionary)


"""create a corpus based on the entire document"""
corpus = create_corpus(indexDictionary, processedText)


"""Create TF-IDF model based on corpus"""
tfIdfModel = create_tfIdf_model(corpus)

"""create TF-IDF weighted"""
tfIdfWeightedCorpus = apply_tfidf_model(tfIdfModel, corpus)


"""create LSI model based on the corpus with TF-IDF weightes and indexDictionary with 100 topics"""
lsiModel = create_lsi_model(tfIdfWeightedCorpus, indexDictionary, numTopics=100)


"""create latent 100d space based on TF-IDF weighted corpus:"""
lsiCorpus = create_latent_space(tfIdfWeightedCorpus, lsiModel)


"""Create similarity matrix for TF-IDF weighted documents and queries:"""
tfIdf_index = create_TFIDS_similarity_matrix(tfIdfWeightedCorpus)


"""create similarity matrix for LSI"""
lsi_index = create_LSI_sim_matrix(lsiCorpus)


"""process query. remove punctuation, convert to lower case, tokenise, stem"""
query = process_query("What is the function of money")


"""convert query to sparse vector"""
query_vec = convert_to_vec(indexDictionary, query)


"""Weight query with TF-IDF weights"""
tfidf_query = tfIdfModel[query_vec]


"""find 3 most relevant documents for TF-IDF weighted query"""
tfIdf_sims = sorted(list(enumerate(tfIdf_index[tfidf_query])),
                    key=lambda t: t[1], reverse=True)[:3]



"""print first 3 topics for LSI-model:"""
print("---first 3 topics for LSI-model---\n")
for topic in lsiModel.show_topics()[:3]:
    print(topic)

print('\n\n')


"""print TF-IDF weights for query:"""
print("---TF-IDF weights for query 'What is the function of money?---\n")
for word in tfidf_query:
    print(indexDictionary.id2token[word[0]] + ': ' + str(word[1]))

print('\n\n')


"""print 3 most relevant documents for TF-IDF weighted query:"""
print("----3 most relevant documents for TF-IDF weighted----\n----query 'What is the function of money?'----\n")

print_3_most_relevant_tfidf(tfIdf_sims, originalParagraphs)
print('\n\n')



"""convert query to latent 100-D space with LSI model:"""
lsi_query = lsiModel[tfidf_query]


"""find top 3 toopics for the query:"""
top_query_topics = sorted(lsi_query, key=lambda t: -abs(t[1]))[:3]


"""print top 3 topics for query:"""
print("----top 3 topics for LSI weighted query----\n----'What is the function of money?'----\n")

for topic in top_query_topics:
    print("Topic" + str(topic[0]) + ":")
    print(lsiModel.print_topic((topic[0])))
    print('\n')
print('\n\n')


"""print 3 most relevant documents for LSI weighted query:"""
doc2similarity = list(enumerate(lsi_index[lsi_query]))
most_relevant_documents_LSI = sorted(doc2similarity, key=lambda t: -t[1])[:3]

print("----Top 3 paragraphs for LSI weighted query----\n----'what is the function of money?'----\n")

for doc_lsi_tuple in most_relevant_documents_LSI:
    print("----paragraph " + str(doc_lsi_tuple[0]) + "----\n")
    paragraph = originalParagraphs[doc_lsi_tuple[0]].split('\r\n')
    i = 0
    while paragraph and i < 5:
        print(paragraph.pop(0))
        i+=1
    print('\n\n')
