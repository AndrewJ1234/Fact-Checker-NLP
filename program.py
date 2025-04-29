import collections

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
            abstract_idf[key][key1] = math.log(words[key1]+1)*idf(key1,docs)
def idf(word,content):
    numOccurences = 0
    if isinstance(content,dict):
        for document in content.values():
            if document.count(word)>0:
                numOccurences+=1
    else:
        for sentence in content:
            if sentence.count(word)>0:
                numOccurences+=1
    return math.log(len(content)/(numOccurences+1))
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
            words[word] = [1,idf(word,docs)]
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

dictAnswer = dict()
def answer_queries(doc):
    with open(doc, 'r', encoding='utf-8') as file:
        text = file.readlines()
    queries = dict()
    key = 1
    for line in text:
        if len(line)>1:
            statement = ""
            result = ""
            line = line.replace("\n"," ").lower()
            if doc=='Training':
                result, statement = line.split(' ',1)
            else:
                statement = line
            queries[key] = statement
            dictAnswer[key] = result
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
    queriesg = queries
    return
def sentence_create_idf(sentence,sentences):
    sentence = sentence.split(' ')
    res = dict()
    word_frequency = dict()
    word_idf = dict()
    for word in sentence:
        if word not in stoplist and word!=':' and word!='"':
            word_frequency[word] = word_frequency.get(word,0)+1
            if word not in word_idf:
                word_idf[word] = idf(word,sentences)
    for word in word_frequency:
        res[word] = (word_frequency[word]*word_idf[word])
    return res
def process_sentences(content,statementKey):
    content = content.replace('\n'," ").lower()
    content_sentences = content.split(".")
    sentences = set(content_sentences)
    key = 1
    sentence_tfidf = collections.defaultdict(list)
    for sentence in sentences:
        sentence_tfidf[key] = sentence_create_idf(sentence,sentences)
        key+=1
    statement_tfidf = sentence_create_idf(queriesg[statementKey],sentences)
    cosines = []
    for key in sentence_tfidf:
        cosines.append([calc_cosine(sentence_tfidf,statement_tfidf[key]),content_sentences[key-1]])
    cosinesorted = sorted(cosines, key=lambda x: x[0], reverse=True)
    top3Article = []
    count = 0
    while len(cosinesorted)-1>count and count<3 :
        count+=1
        top3Article.append(cosinesorted[count])
    return top3Article


queriesg = dict()
top3 = collections.defaultdict(list)
def fetchtop3sentences():
    for key in correct_document:
        top3[key] = process_sentences(correct_document[key],key)
def find_cosine(cosinesAndTF):
    threshold = 0
    acc = 0
    listDeci = [i*.01 for i in range(0,101)]
    for i in listDeci:
        numCorrect = 0
        total = len(cosinesAndTF)
        for cosine,tf in cosinesAndTF:
            pred = 'True' if cosine>=i else 'False'
            if pred==tf:
                numCorrect+=1
        c_acc = numCorrect/total
        if c_acc>acc:
            acc = c_acc
            threshold = i
    return [threshold,acc]
universalCosine = []
def train():
    res = []
    for key in top3:
       res.append([top3[key][0][0],dictAnswer[key]])
    correct_threshold = find_cosine(res)
    universalCosine.append(correct_threshold)
    return
def calculate():
    with open("output.txt",'w',encoding='utf-8') as file:
        for key in top3:
            if top3[key][0][0]>universalCosine[0]:
                file.write("True")
            else:
                file.write("False")
            file.write("\n\n")
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("training")
    p.add_argument("test")
    args = p.parse_args()
    process_documents()
    answer_queries(args.training)
    #ir function is completed, to access corresponding document with each statement, access correct_document dictionary
    #correct_document[{statement number}] = article
    #next step is to fetch top 3 context inside each article
    fetchtop3sentences()
    #fetching top 3 context completed, stored in top3. top3[{statement  number}] = [{list of top 3 sentences inside article}]
    #all that is left is determining T/F based off of the context
    train()
    answer_queries(args.test)
    fetchtop3sentences()
    calculate()