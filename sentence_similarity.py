# Sentence similarity and word similarity

# Global variables:
# data                  Stores numerical values for the json dump
# similarity_values     Stores similarity values of word pairs for json dump
#import time
from nltk.corpus import wordnet as wn
#a= time.time()
from itertools import izip_longest
from pywsd import disambiguate
from semantic_nets_corpus_statistics import length_between_synsets, hierarchical_distance
from frequency_in_wn import return_max_frequency_synset
import numpy as np
import json
#from negatives import process_negation
from length_grammar import dependency_parser

data = {}
similarity_values = {}
div_index = 0


def grouper(iterable, n,fillvalue=None):
    """
    grouper is used for reading n lines from a file altogether
    We use n=3 to read 3 lines from the file containing sentences examples and their previously established similarity value
    :param iterable: file with examples
    :param n: number of lines to be read simultaneously
    :param fillvalue: separator
    :return: n number of strings
    """
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)


def tag_tokens(s):
    """
    Disambiguate the sentence and form a list of tagged words
    :param s: sentence
    :return:  list of words with appropriate synset and part of speech
    """
    list_of_tagged_tokens = disambiguate(s)
    list_of_tagged_tokens = [i for i in list_of_tagged_tokens if i[1] is not None]
    # print list_of_tagged_tokens
    #for ele in list_of_tagged_tokens:
        #print ele[1],":",ele[1].definition()
    #print "\n\n"
    #print type(list_of_tagged_tokens[0])
    #print list_of_tagged_tokens
    return list_of_tagged_tokens

def max_freq_info(list_of_tagged_tokens):
    """
    Get the synset having maximum frequency in a corpus, in this case corpus is wordnet
    :param list_of_tagged_tokens: tokens(tagged words) from tag_tokens
    :return: list of words and corresponding synset with maximum frequency sense from corpus
    """
    tagged_frequency_list = []
    for token in list_of_tagged_tokens:
        pos = str(token[1]).split('.')[1]
        # print token
        max_freq = return_max_frequency_synset(token[0], pos)
        tagged_frequency_list.append([token, max_freq])
    return tagged_frequency_list

def word_similarity(w1, w2):
    """
    Word similarity between words w1 and w2, returns a numeric value between 0 and 1
    :param w1: word 1
    :param w2: word 2
    :return: semantic similarity between word 1 and word 2
    """
    #print w1,w2,length_between_synsets(w1,w2)
    #print w1,w2,hierarchical_distance(w1,w2)
    return length_between_synsets(w1, w2) * hierarchical_distance(w1, w2)


def form_value_vector(d1, d2):
    """
    form a semantic vector for sentences, d1 and d2 are the list of tagged words for s1 and s2 respectively
    :param d1:
    :param d2:
    :return:
    """
    global similarity_values
    global data
    global div_index
    a = len(d1)
    b = len(d2)
    length = max(a, b)
    div_index = min(a, b)
    avg = 0
    i = 0
    j = 0
    semantic_vector = np.zeros(length)
    semantic_vector.fill(0)
    disambiguate_similarity=0.0
    max_frequency_similarity=0.0
    # print semantic_vector
    for w1 in d1:
        previous_avg=0
        for w2 in d2:
            # print tuple[0][1],tuple[1] # tuple[0][1] is disambiguate and tuple[1] is maximum frequency tuple
            # print "Words:",w1[0][0],w2[0][0]
            if ".n." in str(w1[0][1]) and ".n." in str(w2[0][1]):
                disambiguate_similarity = word_similarity(w1[0][1], w2[0][1])
                #print "DIS",disambiguate_similarity
                max_frequency_similarity = word_similarity(w1[1], w2[1])
                #print "freq",max_frequency_similarity
                #previous_avg = (disambiguate_similarity + max_frequency_similarity) / 2.0
            if ".v." in str(w1[0][1]) and ".v." in str(w2[0][1]):
                disambiguate_similarity = word_similarity(w1[0][1], w2[0][1])
                max_frequency_similarity = word_similarity(w1[1], w2[1])
                #previous_avg = (disambiguate_similarity + max_frequency_similarity) / 2.0
            key = w1[0][0], w2[0][0]
            previous_key = ' '.join(key)

            if disambiguate_similarity == 0.0:
                previous_avg = max_frequency_similarity
            elif max_frequency_similarity == 0.0:
                previous_avg = disambiguate_similarity
                # elif abs(disambiguate_similarity - max_frequency_similarity) > 0.2:
                # previous_avg = max(disambiguate_similarity, max_frequency_similarity)
            else:
                previous_avg = (disambiguate_similarity + max_frequency_similarity) / 2.0
            # print "sim:",previous_avg
            # print previous_avg

            data[previous_key] = previous_avg
            similarity_values = json.dumps(data)

            if avg < previous_avg:
                avg = previous_avg
                # new_key=previous_key
        # print "needed value:",avg
        # print "word pair:", new_key
        print w1,w2
        print "avg", avg
        semantic_vector[i] = avg
        i = i + 1
        avg = 0
        key = ""
    return semantic_vector


def word_order_sim(vector_1, vector_2):
    l1 = len(vector_1)
    l2 = len(vector_2)
    l = max(l1, l2)
    if l2 > l1:
        order_vector_1 = np.arange(1, l)
        order_vector_2 = np.zeros(l)
    else:
        order_vector_2 = np.arange(1, l)
        order_vector_1 = np.zeros(l)


def similarity(vector_1, vector_2):
    """
    Sentence similarity
    vector_1 and vector_2 are semantic vectors from form_value_vector
    :param vector_1: semantic vector for sentence 1
    :param vector_2: semantic vector for sentence 2
    """
    global similarity_values
    global div_index
    global data
    count = 0
    shift=0
    ###########################################################################################
    # print vector_1
    # print vector_2
    # print np.dot(vector_1,vector_2.T)
    # word_order_index=(1.0-DELTA)*word_order_similarity(s1,s2)
    # print "WOS:",word_order_index

    # print "np.dot(vector_1, vector_2)",np.innera(vector_1, vector_2)
    # print "np.linalg.norm(vector_1):",np.linalg.norm(vector_1)
    # print "np.linalg.norm(vector_2):",np.linalg.norm(vector_2)
    # print "(np.linalg.norm(vector_1) * np.linalg.norm(vector_2)):",(np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
    # print "DELTA*np.dot(vector_1, vector_2):",DELTA*np.dot(vector_1, vector_2)
    # print "ORIGINAL:",np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
    # print vector_1
    # print vector_2
    ############################################################################################
    sim = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
    #print sim
    for index in vector_1:
        if index > 0.80:
            count = count + 1
    for index in vector_2:
        if index > 0.80:
            count = count + 1

    # print "Divided by vec size:",sim/(vector_1.size)
    # print "Divided by vec/2:",sim/((vector_1.size)/2)
    print count
    if count > 2:
        sim = sim / (count / 1.8)
        if sim > 1.0:
            sim=1.0
        #print sim
        #negation = process_negation(s1,s2)
        #print "nenation",negation
        #if negation == 0.0:
            #sim=sim/1.5
            #pass
        #print sim

    else:
        try:
            sim = sim / (float(vector_1.size) / 2)
            #print "inside try",sim
        except:
            sim=sim

    if sim> 1.0:
            sim=1.0
        #negation = process_negation(s1, s2)
        #print "nenation" , negation
        #if negation == 0.0:
            #sim=sim/1.5
            #pass
        #print "after negation:",sim
        #print sim
        # similarity_values.dumps(data)
        #print similarity_values
        # return similarity_values
    #print sim
    length_s1=len(s1.split())
    length_s2=len(s2.split())
    length_difference=abs(length_s1-length_s2)
    if length_difference==0:
        pass
    else:
        try:
            shift=0.10*np.log(length_difference+1)
            #print "shift",shift
        except:
            shift=0.0

    dependency=dependency_parser(s1,s2)
    #print dependency
    if sim>0.85:
        if "no" in s1.split(" ") or "no" in s2.split(" ") or "nobody" in s1.lower().split(" ") or "nobody" in s2.lower().split(" "):
            #print("%.2f" % (sim-dependency-shift))
            sim=sim-dependency-shift
            if shift==0.0:
                sim=0.75

        else:
            #print("%.2f" % (sim-dependency))
            sim = sim - dependency
            pass
    else:
        #print("%.2f" % (sim-shift))
        if sim>0.5:
            sim=sim-shift

    if length_s1==length_s2:
        missing1 = [x for x in s1.split(" ") if x not in s2.split(" ")]
        #print missing1
        missing2 = [x for x in s2.split(" ") if x not in s1.split(" ")]
        #print missing2
        #compare synsets of missing1 and missing2 and if equal then sim=1 else sim=0.7
        if len(missing1) == 1:
            syn1=wn.synsets(missing1[0])
            syn2=wn.synsets(missing2[0])
            for syn1_1 in syn1:
                for syn2_1 in syn2:
                    if word_similarity(syn1_1,syn2_1)>0.8:
                        sim=0.95
                        break
                if sim==0.95:
                    break
                sim=0.7
                if s1!=s2:
                    sim=0.5


    if "no" in s1.split(" ") or "no" in s2.split(" ") or "nobody" in s1.split(" ") or "nobody" in s2.split(" "):
        sim = 0.7-shift
    if s1==s2:
        sim=1.0
    print abs(sim)
    data["sentence_similarity"] = abs(sim)
    similarity_values = json.dumps(data)
    #print similarity_values
    return similarity_values

s1="existing COBOL programs to enhance functionality and/or correct errors in program logic"
s2="sets, logics and functions"
#s1="GPS"
#s2="appropriate"
#with open('asd') as f:
    #for lines in grouper(f, 3, ''):
        #assert len(lines) == 3
    #for line in f:
        #lines=line.split('\t')
        #s1=lines[1].rstrip()
        #print s1
d1 = tag_tokens(s1)
d1 = max_freq_info(d1)
#s2 = lines[2].rstrip()
#print s2
d2 = tag_tokens(s2)
d2 = max_freq_info(d2)
vector_1 = form_value_vector(d1, d2)
print "V1",vector_1
vector_2 = form_value_vector(d2, d1)
print "V2",vector_2
#print lines[1].rstrip()
similarity(vector_1, vector_2)
        #print "\n"
