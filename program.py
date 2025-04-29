from stop_list import closed_class_stop_words
import string
import math
from nltk.stem import WordNetLemmatizer
import nltk
import argparse
nltk.download('wordnet')
stem = WordNetLemmatizer()
stoplist = set(closed_class_stop_words)
keylist = list()
docs = dict()
abstract_idf = dict()
correct_document = dict()
def remove_plural(word):
    if word.endswith('s') and not word.endswith('ss') and len(word) > 3:
        return word[:-1]
    return word
def process_documents():
    with open("articles", 'r', encoding='utf-8') as file:
        text = file.readlines()
    skip = 0
    key = -1
    for line in text:
        line = line.replace('\n',"").lower()
        if len(line)<1:
            continue
        if line.startswith("##"):
            key = int(line[2::])
            docs[key] = ""
        docs[key] +=line
    for key in docs.keys():
        words = dict()
        translator = str.maketrans('', '', string.punctuation)
        querywords = docs[key].translate(translator).split(" ")
        for word in querywords:
            word = stem.lemmatize(word,pos='n')
            word = remove_plural(word)
            if word == '':
                continue
            if word in stoplist:
                continue
            try:
                float(word)
                continue
            except ValueError:
                pass
            if word in words:
                words[word]+= 1
            else:
                words[word] = 1
        abstract_idf[key] = dict()
        for key1 in words.keys():
            abstract_idf[key][key1] = math.log(words[key1]+1)*idf(key1)
def idf(word):
    numDocOccurences = 0
    for document in docs.values():
        if document.count(word)>0:
            numDocOccurences+=1
    return math.log(len(docs)/(numDocOccurences+1))
def process_query(query):
    words = dict()
    translator = str.maketrans('', '', string.punctuation)
    querywords = query.translate(translator).split(" ")
    for word in querywords:
        word = stem.lemmatize(word, pos='n')
        word = remove_plural(word)
        if word == '':
            continue
        if word in stoplist:
            continue
        try:
            float(word)
            continue
        except ValueError:
            pass
        if word in words:
            words[word][0]+=1
        else:
            words[word] = [1,idf(word)]
    idf_tags = dict()
    for key in words.keys():
        idf_tags[key] = math.log(words[key][0]+1)*words[key][1]
    return idf_tags
def calc_cosine(x,y):
    top = 0
    bottom_left = 0
    bottom_right = 0
    for key in x.keys():
        top+= (x[key]*y.get(key,0))
        if key in y.keys():
            bottom_left+=x[key]**2
            bottom_right +=y[key]**2
    bottom = (bottom_left*bottom_right)**.5+1
    return top/bottom


def answer_queries(doc):
    with open(doc, 'r', encoding='utf-8') as file:
        text = file.readlines()
    queries = dict()
    key = 1
    for line in text:
        if len(line)>1:
            statement = ""
            line = line.replace("\n"," ").lower()
            if doc=='Training':
                result, statement = line.split(' ',1)
            else:
                statement = line
            queries[key] = statement
            keylist.append(key)
            key+=1
    count = 0
    for key in keylist:
        count+=1
        vector = process_query(queries[key])
        cosines = []
        for key1 in abstract_idf.keys():
            cosines.append([key1,calc_cosine(vector,abstract_idf[key1])])
        cosinesorted = sorted(cosines, key=lambda x: x[1], reverse=True)
        correct_document[key] = docs[cosinesorted[0][0]]
    return
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("file")
    args = p.parse_args()
    process_documents()
    answer_queries(args.file)