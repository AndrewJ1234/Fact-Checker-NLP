import collections

from stop_list import closed_class_stop_words,negatives
import string
import math
from nltk.stem import WordNetLemmatizer
import nltk
import argparse
nltk.download('wordnet')
stem = WordNetLemmatizer()
stoplist = set(closed_class_stop_words)
neglist = set(negatives)
keylist = list()
docs = dict()
abstract_idf = dict()
correct_document = dict()
docs_tokenized = dict()
def remove_plural(word):
    if word.endswith('s') and not word.endswith('ss') and len(word) > 3:
        return word[:-1]
    return word
def process_documents():
    # with open("articles.txt", 'r', encoding='utf-8') as file:
    with open("mod_articles.txt", 'r', encoding='utf-8') as file:
        text = file.readlines()
    key = None
    for line in text:
        line = line.replace('\n',"").lower()
        if len(line)<1:
            continue
        line = line.lstrip()
        line = line.replace('\ufeff','')
        if line.startswith("##"):
            key = int(line[2::])
            docs[key] = ""
            continue
        docs[key] +=line
    for key in docs:
        tokens = docs[key].translate(str.maketrans('', '', string.punctuation)).lower().split()
        tokens = [remove_plural(stem.lemmatize(w, pos='n')) for w in tokens]
        docs_tokenized[key] = tokens
    for key in docs.keys():
        words = dict()
        translator = str.maketrans('', '', string.punctuation)
        querywords = docs[key].translate(translator).split(" ")
        for word in querywords:
            word = stem.lemmatize(word,pos='n')
            word = remove_plural(word)
            word = word.replace('"','')
            word = word.replace(':','')
            word = word.replace('(','')
            word = word.replace(')','')
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
    numOccurences = 0
    for document in docs_tokenized.values():
        if word in document:
            numOccurences += 1
    if numOccurences == 0:
        return 0
    return math.log(len(docs_tokenized) / numOccurences)
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
        word = word.replace('"', '')
        word = word.replace(':', '')
        word = word.replace('(', '')
        word = word.replace(')', '')
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

dictAnswer = dict()
def answer_queries(doc):
    global queriesg
    keylist.clear()
    dictAnswer.clear()
    correct_document.clear()
    with open(doc, 'r', encoding='utf-8') as file:
        text = file.readlines()
    queries = dict()
    key = 1
    for line in text:
        if len(line)>1:
            statement = ""
            result = ""
            line = line.replace("\n"," ").lower()
            if doc=='Training.txt':
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

        correct_document[key] = [docs[cosinesorted[0][0]],cosinesorted[0][0]]
    with open('output2.txt','w',encoding='utf-8') as file:
        for key in correct_document:
            file.write(queries[key])
            file.write('\n'+str(correct_document[key][1]))
            file.write('\n\n')
    queriesg = queries
    return
def calcNegatives(sentence):
    num_negatives = 0
    num_words =0
    for word in sentence:
        if word in negatives:
            num_negatives+=1
        num_words+=1
    return num_negatives/num_words
def find_negative(negative_result,flagari):
    best_threshold = 0
    best_f1 = 0
    scale = 0
    if flagari == 1:
        scale = 1
    else:
        scale = .01
    for threshold in [i * scale for i in range(101)]:
        TP = FP = FN = 0
        for score, label in negative_result:
            actual = label.lower()
            predicted = 'false' if score > threshold else 'true'

            if predicted == 'false':
                if actual == 'false':
                    TP += 1
                else:
                    FP += 1
            elif predicted == 'true' and actual == 'false':
                FN += 1
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold
universalNegative = []
def ari(key):
    words = docs_tokenized[correct_document[key][1]]
    num_words = len(words)
    num_chars = 0
    for word in words:
        num_chars+=len(word)
    text = docs[correct_document[key][1]]
    num_sentences = len(text.split('.'))
    if num_sentences ==0 or num_words ==0:
        return 0
    aris = 4.71*(num_chars/num_words)+.5*(num_words/num_sentences)-21.43
    return aris
def train_ari():
    arilist = []
    for key in correct_document:
        ari_num = ari(key)
        result = dictAnswer[key]
        arilist.append([ari_num,result.lower()])
    threshold = find_negative(arilist,1)
    universalNegative.append(threshold)
def train_negatives():
    res = []
    for key in negative_words:
        res.append([negative_words[key], dictAnswer[key]])
    correct_threshold=find_negative(res,0)
    universalNegative.append(correct_threshold)
def calculate():
    with open("output.txt",'w',encoding='utf-8') as file:
        for key in negative_words:
            num_negatives = negative_words[key]
            if num_negatives>universalNegative[1]:
                file.write("FALSE")
            else:
                file.write("TRUE")
            file.write("\n\n")
def baseline():
    with open("ari_baseline_output.txt", 'w', encoding='utf-8') as file:
        for key in correct_document:
            score = ari(key)
            if score > universalNegative[1]:
                file.write("FALSE")
            else:
                file.write("TRUE")
            file.write("\n\n")
negative_words = {}
def find_negatives(article):
    words = article.split(' ')
    num_words = 0
    num_negatives = 0
    for word in words:
        word = word.replace('.','')
        word = word.replace('"', '')
        word = word.replace(':', '')
        word = word.replace('(', '')
        word = word.replace(')', '')
        if len(word)>0:
            if word in neglist:
                num_negatives+=1
            num_words+=1
    return num_negatives/num_words

def calculate_negative_article():
    for key in correct_document:
        article = correct_document[key][0]
        score = find_negatives(article)
        negative_words[key] = score
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("training")
    p.add_argument("test")
    args = p.parse_args()
    process_documents() #creates tfidf vector for each article in abstract_idf
    answer_queries(args.training) #takes in the statement files, calculates tfidf of statement, compare with abstract_idf to find highest matching and assign to correct_document[key] = article
    calculate_negative_article()
    #train()
    train_ari()
    train_negatives()

    negative_words.clear()
    answer_queries(args.test)
    #fetchtop3sentences()
    calculate_negative_article()
    calculate()
    baseline()