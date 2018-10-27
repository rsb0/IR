import random; random.seed(123)
import codecs
import os
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import string

def load_and_process():
    f = codecs.open("pg3300.txt", "r", "utf-8")

    punct = string.punctuation+"\r\n\t"
    paragraph = ""
    paragraphList=[]
    stemmer = PorterStemmer()

    """boolean value used to detect several newline characters in a row"""
    prevLineNewline = False

    """Loop over f.readlines(). If element is not a single newline character: add the element to the paragraph string.
    If element is a single newline character: remove punctuation + whitespace characters from paragraph string
    and add paragraph to paragraphList"""
    for element in f.readlines():
        if element == "\r\n":
            if not prevLineNewline:
                prevLineNewline = True
                paragraph = paragraph.translate({ord(c): None for c in punct})
                paragraphList.append(paragraph.strip())
                paragraph = ""
            else:
                pass
        else:
            paragraph += element + " "
            if prevLineNewline:
                prevLineNewline = False

    f.close()


    """loop over paragraphList and remove all elements containing the word "Gutenberg" """
    excludeGutenbergList = []
    for element in paragraphList:
        if "Gutenberg" not in element:
            if "GUTENBERG" not in element:
                excludeGutenbergList.append(element)
    originalParagrahpList = excludeGutenbergList


    """tokenise paragraph, remove space from end of strings, convert to lower case, stem words and add to processedList"""
    processedList = []
    for element in excludeGutenbergList:
        tokenisedParagraph = element.lower().split(" ")
        tokenisedParagraphList = []
        for word in tokenisedParagraph:
            tokenisedParagraphList.append(stemmer.stem(word))
        processedList.append(tokenisedParagraphList)


    return processedList, originalParagrahpList

processedList, originalParagraphList = load_and_process()
