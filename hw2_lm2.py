import numpy as np
import re
from collections import Counter 
from functools import reduce
import matplotlib.pyplot as plt

class LanguageModel:
    
    def __init__(self, n_gram, is_laplace_smoothing, backoff = None):
        
        self.n_gram = n_gram
        self.is_laplace_smoothing = is_laplace_smoothing
        self.backoff = backoff
        self.n_minus_one = n_gram-1
    ####################################
    
    def train(self, training_file_path):
        
        
        n_m = self.n_minus_one
        
        def helper(wordList, n): #helper function to make file sentences into n_grams, returns dictionary with key as n_gram and value as count
            ngrams = zip(*[wordList[i:] for i in range(n)])
            possibilities = {}
            for ng in ngrams:
                if ng not in possibilities:
                    possibilities[ng] = 1
                else:
                    possibilities[ng] += 1
            return possibilities
        
        
        
        f = open(training_file_path, "r")
        words = []#list of all words
        tmp = []
        for line in f:
            w = line.split()    
            for token in w:
                words.append(token) 
        
        position=0     
        for w in (words):
            if (words.count(w)==1): #frequency of word is one
                words[position] = "<unk>" #replace with <unk> token
            position+=1
        
        v = len(Counter(words).keys()) #V is the size of the vocabulary which is the number of unique unigrams.
        self.vocab = v
                
        
        dictionary=helper(words, self.n_gram)
        
        if(self.n_gram > 1):
            dictionaryn_m = helper(words, n_m) #need n-1_gram dictionary for 2+ gram to calculate "history"
        #P(n_gram) = (word | history)
        
        self.nn = tokens = sum(dictionary.values())
        #Re-assign the value of keys to the probability WITHOUT add-1 Smoothing count/number of last word
        if (self.is_laplace_smoothing == False):
            #different for unigrams
            if (self.n_gram == 1):
                
                tokens = sum(dictionary.values()) #total number of word tokens (n, where value is count of n_gram)
                
                
                for key in dictionary.keys():
                    count = dictionary[key]
                    dictionary[key] = (count/tokens) #re-assign value to be the probability of the key (unigram)
                
                

            else: #case for not unigram
                for key in dictionary.keys():
   
                    
                    last = key[:n_m] #access the "history"
      
                    count = dictionary[key] #count of n_gram
        
                    try:
#                         dictionary[key] = (count/float(words.count(last))) 
                        dictionary[key] = (count/dictionaryn_m[last]) #re-assign value to: ngram count/ ngram-1 count (probability of the n_gram)
                    except Exception as e: 

                        print(last, type(last), type(key), key)
                        
        #Re-assign the value of keys to the probability WITH add-1 Smoothing count/number of last word               
        else:
            
            
            if (self.n_gram == 1):
                tokens = sum(dictionary.values()) #"N"
                
                
                for key in dictionary.keys():
                    count = dictionary[key] #count of the unigram
                    dictionary[key] = ((count+1)/(tokens+v)) #Re-assign unigram count to it's probability: add one to unigram count, divide by N+V (laplace smoothing)
                #print(tokens, v)
            else:
                for key in dictionary.keys(): #re-assigning n_grams value

                    
                    
                    
                    last = key[:n_m] #make the "history"




                    count = dictionary[key] #the n_gram count
                    try:
                        dictionary[key] = ((count+1)/(dictionaryn_m[last]+v)) #re-assign to it's probability: count of (n_gram+1)/count of n-1_gram+V
                        
                    except Exception as e: 

                        print(tuple(last), words.count("<unk>"))
            
        self.d = dictionary
        if(self.n_gram > 1): #storing n-1_gram
            self.dn_m = dictionaryn_m
            
        self.unique = words
        #print(dictionary)

    ####################################
    def generate(self, num_sentences):
        di = self.d
        startD = {}
        sentences=[]
        
        def newD(dicti, word): #helper function to make subdictionaries of n_grams that all begin with a certain word
            newD={}
            for key in dicti.keys(): 
                if(key[0] == word):
                    newD.update({key: dicti[key]})
            return(newD) #returns a dictionary that only has keys that begind with the passed in word

        if (self.n_gram !=1):        
            startD = newD(di, '<s>') #making dictionary of words starting with <s>
            p = list(startD.values()) #grab the list of values (probabilities from dictionary of values starting with <s>)
            p = [x/sum(p) for x in p] # NORMALIZING the probability values

            for i in range(num_sentences):
                sentence =""
                #print one ngram from startD (with np.random.choice)
                z = np.random.choice((len(startD)), p=p) #randomly select an index of an n_gram through their corresponding probabilities (list of p)
                t = list(startD)[z] #grab the associated n_gram from that index
                #print(t)
                sentence = str(t) #typecast to a string
                
                l=True #begin our n_gram adder while loop
                while(l):
                    if('</s>' not in t): #check every iteration if </s> in n_gram. (if yes, then sentence will end, a.k.a while loop will end)
                        prevWord = t[-1] #grabbing previous word so that we can make a dictionary of only n_grams beginning with that word
                        dd = newD(di, prevWord) #create the dictionary ^
                        #print one ngram from newD (with np.random.choice)
                        pp = list(dd.values()) #list of the probabilities of each n_gram begining with previous word
                        pp = [x/sum(pp) for x in pp] #NORMALIZING the probabilities
                        zz = np.random.choice((len(dd)), p=pp) #using random generation to grab an index accordingly
                        t = list(dd)[zz] #grab the associated n_gram
                        if (t != '<s>'): #as long as <s> is NOT contained in the n_gram, then we will append the selected n_gram to our sentence
                            #print(t)
                            
                            sentence+= str(t[1:])
                    else:
                        l=False #to end while loop once </s> is encountered
                #below is just to remove the tuple punctuations
                sentence = sentence.replace('(\'', "")
                sentence = sentence.replace('\',)', " ")
                sentence = sentence.replace('("', "")
                sentence = sentence.replace('",)', " ")
                sentence = sentence.replace('\',', "")
                sentence = sentence.replace('\')', " ")
                sentence = sentence.replace('\"', "")
                #recent edits
                sentence = sentence.replace(' \'', " ")
                sentence = sentence.replace('"', "")
                sentence = sentence.replace(' <s> ', "")
                sentence = sentence.replace(')', " ")
                sentence = sentence.replace(',', "")
                if(sentence[-1:] == " "):
                    sentence = sentence[:-1]

                
                
                #sentence = sentence[:-1]
                sentences.append(sentence)  #to return a LIST of sentences, we will append all of the sentences      
                    
        else: #unigram sentence generation

   
            p = list(di.values()) #list for the probabilities of ALL unigrams 
            p = [x/sum(p) for x in p] #NORMALIZING
            
            for i in range(num_sentences):
                sentence =""
                #print((('<s>',)))
                sentence = "<s> " #Manually start sentence with <s>
                t=''
                while(t!=('</s>',)): #continue until encounter </s>
                    z = np.random.choice((len(di)), p=p) #randomly select associated index
                    t = list(di)[z] #grab that unigram of associated index
                    if (t != (('<s>',))): #as long as it is not a <s> token
                        #print(t)
                        sentence+= str(t) #add it to our sentence
                #below is just to remove the tuple punctuations
                sentence = sentence.replace('(\'', "")
                sentence = sentence.replace('\',)', " ")
                sentence = sentence.replace('("', "")
                sentence = sentence.replace('",)', "")
                if(sentence[-1:] == " "):
                    sentence = sentence[:-1]
                sentences.append(sentence)  
        
        
        #return a list of strings generated using Shannon's method of length num_sentences
        return(sentences) #a list of sentences
                

    ###################################
    def score(self, sentence): #TODO: Convert sentence to have <unk> tokens BEFORE ANYTHING
        
        
        lst = []
        di = self.d
        tokens = self.nn
        
        if(self.n_gram > 1): #this is to access dictionary of n-1_gram we need to access IF an n_gram does not exist in our training data for Laplace smoothing: P = (count(n_gram)=0)+1/count(n-1_gram)+V
            dn_m = self.dn_m
           
        unkvalue = None #initially setting <unk> to None
        unk = [value for key, value in di.items() if '<unk>' in key] #variable unk = the PROBABILITY of the <unk> token in our dictioanry(gonna be a list, which we can use to our advantage to see if the len(list) if >0, because if NOT that means our training data had no <unk> variables)
        if (len(unk)!=0): #if unk list contains the unk probability value
            unkvalue = unk[0] #assign unkvalue to that number
            #print(unkvalue)
        
        
        v = self.vocab
        unique = self.unique #list of all unique words (includes <unk>)
        sentence = sentence.split(' ') #must to this to create n_grams

        unique_words = unique # list of all unique words (includes <unk>)
        u = list(sentence) #converting our parameter sentence into a list
        
        def histogram(li):
            a = plt.hist(x=lst, bins='auto', color='#0504aa')
            plt.xlabel('Probabilities')
            plt.ylabel('Frequencies')
            plt.title('Probability and Frequency Values of n_gram')

            return(a)



        #go through all the words in sentence and see if they are in our training set
        #ONLY DO THIS IF UNK EXISTS IN DICTIONARY ALREADY
        if (unkvalue!=None):
            for i in range (len(u)):
                if (u[i] not in unique): #if a certain word is NOT in our training and our training has an <unk> token, then we will replace that word with <unk> in our sentence
                    u[i] = '<unk>'

        def listToString(s):  #helper function that will convert our list back into a sentence
            # initialize an empty string 
            str1 = " " 

            # return string   
            return (str1.join(s)) 

        sentence = listToString(u) #convert list into new sentence
        sentence = sentence.split(' ') #need to do this to create our n_grams, splitting on the space
        
        def helper(s, n): #helper function (same as above) to create n_grams
            ngrams = zip(*[s[i:] for i in range(n)])
            possibilities = {}
            for ng in ngrams:
                if ng not in possibilities:
                    possibilities[ng] = 1
                else:
                    possibilities[ng] += 1
            return possibilities


        
        if(self.is_laplace_smoothing == False):
            if (self.n_gram == 1): #for unigrams, we need to just return the value of e/ unigram in the dictionary
                #lst = []
                
                sentence_ng = helper(sentence, self.n_gram) #convert sentence into unigrams, pass e/ into dictionary with counts as value
                
                for key in sentence_ng.keys():
                    
                    if (key in di.keys()): #if unigram exists in Training
                        lst.append(di[key]) #append it's probability to a list

                    else: #unigram does NOT exist in our training data so...
                        
                        if (unkvalue != None): #we need to see IF <unk> exists in our training data, if so, we append the probability value of the <unk> token
                            lst.append(unkvalue) 
                        else: #training does NOT have <unk> token, so the p(unigram) = 0, making the whole sentence 0!
                            lst.append(0)

            
                return(reduce(lambda x, y: x*y, lst)) #multiply all the values together (this is the equation from text)
                
        
                


            else:
                #if n_gram
                #lst = []
                
                sentence_ng = helper(sentence, self.n_gram) #make n_gram out of the sentence passed in
                
                for key in sentence_ng.keys():
                    if (key in di.keys()): #if n_gram exists in Training
                        countn_gram = di[key] #this is the probability of it occuring

                        lst.append(countn_gram) #append to it's probability value to list
                        
                    else: #if the n-gram does NOT occur in our training, we simply append 0, because laplace smoothing is off
                        lst.append(0)
                        
                                        
                if (reduce(lambda x, y: x*y, lst)==0):
                    return(0)
                else:
                    return(reduce(lambda x, y: x*y, lst)) #multiply all the values of the n_gram together
                        
        
        #if Laplace is True
        else:
            #SAME AS ABOVE SITUATION EXCEPT AT LEAST = 1/N+V BECAUSE OF LAPLACE INSTEAD OF 0
            if (self.n_gram == 1): #for unigrams, we need to just return the value in the dictionary

                #lst = []
                sentence_ng = helper(sentence, self.n_gram)
                
                for key in sentence_ng.keys():
                    
                    if (key in di.keys()): #exists in Training
                        lst.append(di[key])
                        
                        
                    else: #not in training 
                        if (unkvalue != None): #training has <unk> token
                            lst.append(unkvalue)
                            
                        else: #training does NOT have <unk> token and the unigram is NOT in our training data
                            lst.append((1/(tokens+v))) #we append 1/n+v
                            
            
                return(reduce(lambda x, y: x*y, lst)) #multiply all the values to eachother, no 0 case because laplace smoothing
                    
            else: #ngrams and laplace smoothing
                #if n_gram
                #lst = []
                
            #SAME AS ABOVE SITUATION EXCEPT AT LEAST = 1/N+V BECAUSE OF LAPLACE INSTEAD OF 0

                sentence_ng = helper(sentence, self.n_gram) #create n_gram out of sentence put into dictionary: key, value
                
                for key in sentence_ng.keys():
                    if (key in di.keys()): #if n_gram exists in Training data dictionary
                        countn_gram = di[key] #this is the probability of it occuring

                        lst.append(countn_gram) #append to list
                        
                    else: #does not exist, we need to access our n-1_gram dictionary dn_m
                        if (key[:self.n_minus_one] in dn_m.keys()): #cropping the n_gram to n-1_gram, seeing if it is in the n-1_gram dictionary
                            countn_mgram = dn_m[key[:self.n_minus_one]] #saving the count of n-1_gram to save as denominator
                            lst.append((1/(countn_mgram+v))) #appending (count(n_gram)=0)+1/count(n-1_gram+v)
                            
                        else: #if n-1_gram is not in our training, we will just have v as the denominator  ((count(n_gram)=0)+1/(count(n-1_gram)=0)+v))
                            lst.append((1/v))  #append it to our list
                        
                if (reduce(lambda x, y: x*y, lst)==0):
                    return(0)
                else:
                    return(reduce(lambda x, y: x*y, lst)) #multiple all the probabilities together
        
        

    
    