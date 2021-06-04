# Sentence similarity and word similarity

# Global variables:
# data                  Stores numerical values for the json dump
# similarity_values     Stores similarity values of word pairs for json dump
from itertools import izip_longest
from pywsd import disambiguate
from semantic_nets_corpus_statistics import length_between_synsets, hierarchical_distance, word_order_similarity
from frequency_in_wn import return_max_frequency_synset
from return_syn_lo import compare_def_synset, synset_lookup
from nltk import word_tokenize
import numpy as np
import json

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
    l=[]
    tup=()
    list_of_tagged_tokens = word_tokenize(s)
    for ele in list_of_tagged_tokens:
        #print compare_def_synset(ele) # here we get synset and we have format in the original format, form a tuple word-synset
	    syn=compare_def_synset(ele)
	#print ele
	#print type(syn)
	    tup=(ele,syn)
	#print tup
	    l.append(tup)
    #print l
    l = [i for i in l if i[1] is not None]
    #print "NEW",l
    return l



def word_similarity(w1, w2):
    """
    Word similarity between words w1 and w2, returns a numeric value between 0 and 1
    :param w1: word 1
    :param w2: word 2
    :return: semantic similarity between word 1 and word 2
    """
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
    # print semantic_vector
    for w1 in d1:
        for w2 in d2:
            # print tuple[0][1],tuple[1] # tuple[0][1] is disambiguate and tuple[1] is maximum frequency tuple
            # print "Words:",w1[0][0],w2[0][0]

            disambiguate_similarity = word_similarity(w1[1], w2[1])
            # print "DIS",disambiguate_similarity
            #max_frequency_similarity = word_similarity(w1[1], w2[1])
            # print "freq",max_frequency_similarity
            #previous_avg = (disambiguate_similarity + max_frequency_similarity) / 2.0

            key = w1[0][0], w2[0][0]
            previous_key = ' '.join(key)

            if disambiguate_similarity == 0.0:
                previous_avg = disambiguate_similarity
            elif disambiguate_similarity == 0.0:
                previous_avg = disambiguate_similarity
                # elif abs(disambiguate_similarity - max_frequency_similarity) > 0.2:
                # previous_avg = max(disambiguate_similarity, max_frequency_similarity)
            else:
                previous_avg = (disambiguate_similarity + disambiguate_similarity) / 2.0
            # print "sim:",previous_avg
            # print previous_avg

            data[previous_key] = previous_avg
            similarity_values = json.dumps(data)

            if avg < previous_avg:
                avg = previous_avg
                # new_key=previous_key
        # print "needed value:",avg
        # print "word pair:", new_key
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

    for index in vector_1:
        if index > 0.80:
            count = count + 1
    for index in vector_2:
        if index > 0.80:
            count = count + 1

    # print "Divided by vec size:",sim/(vector_1.size)
    # print "Divided by vec/2:",sim/((vector_1.size)/2)
    if count > 2:
        #print "IN 1"
        sim = sim / (count / 1.8)
        print sim
        data["sentence_similarity"] = sim
        similarity_values = json.dumps(data)
    else:
        #print count
        #print "IN 2"
        #print sim
        sim = sim / (vector_1.size /2 )
        print sim
        data["sentence_similarity"] = sim
        similarity_values = json.dumps(data)
        # similarity_values.dumps(data)
        # print similarity_values
        # return similarity_values


with open('sample_sentence') as f:
    for lines in grouper(f, 3, ''):
        assert len(lines) == 3
        s1 = lines[0].rstrip()
        print s1
        d1 = tag_tokens(s1)
        #print "D1",d1
        #max_freq_info(d1)
        s2 = lines[2].rstrip()
        print s2
        d2 = tag_tokens(s2)
        #d2 = max_freq_info(d2)
        vector_1 = form_value_vector(d1, d2)
        # print "V1",vector_1
        vector_2 = form_value_vector(d2, d1)
        # print "V2",vector_2
        # print lines[1].rstrip()
        similarity(vector_1, vector_2)
        print "\n"
