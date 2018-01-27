from textblob.classifiers import NaiveBayesClassifier
import os,glob,re
from string import punctuation
from collections import defaultdict
from collections import Counter
import numpy as np
from collections import OrderedDict
import copy
from sklearn import metrics
import operator
import pprint


Train_Set_NEG  = []
count_neg = 0
Training_Set = []
ID_val_List  = []
for line in open("hotelNegT-train.txt",'r',encoding="utf8").readlines():
    if re.match(r'ID-[0-9].*',line):
        count_neg+=1
        each_line = re.sub(r'ID-[0-9].*_','',line)
        each_line = each_line.strip("\n")
        Train_Set_NEG.append(each_line[8:])

for each_sentence in Train_Set_NEG:
    Training_Set.append((each_sentence,'NEG'))
    

Train_Set_POS  = []
count_pos = 0
for line in open("hotelPosT-train.txt",'r',encoding="utf8").readlines():
    if re.match(r'ID-[0-9].*',line):
        count_pos+=1
        each_line = re.sub(r'ID-[0-9].*_','',line)
        each_line = each_line.strip()
        Train_Set_POS.append(each_line[8:])

for each_sentence in Train_Set_POS:
    Training_Set.append((each_sentence,'POS'))

classifier = NaiveBayesClassifier(Training_Set)  

Test_Set = []
for line in open("TestSet.txt",'r',encoding="utf8").readlines():
    if re.match(r'ID-[0-9].*',line):
        each_line = re.sub(r'ID-[0-9].*_','',line)
        each_line = each_line.strip()
        Test_Set.append(each_line[8:])
        ID_val_List.append(each_line[0:7])

sol_list = []
for each_item in Test_Set:
    sol_list.append (classifier.classify(each_item))

dictionary = OrderedDict()
dictionary = dict(zip(ID_val_List, sol_list))

with open("Upadhyaya-Swathi-assgn3-out_builtin_Func.txt",'w') as ofile:
        for key,val in dictionary.items():
            ofile.write((str(key) + '\t' + str(val) + '\n'))
