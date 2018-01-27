import os,glob,re
from string import punctuation
from collections import defaultdict
from collections import Counter
import numpy as np
from collections import OrderedDict
import copy
from sklearn import metrics
import operator
from textblob.classifiers import NaiveBayesClassifier

list_negative_word = []
list_positive_word = []
Unique_list_positive_word = []
Unique_list_negative_word = []
cwd = os.getcwd() 
count_positive_ID = 0
count_negative_ID = 0
Negative_Word_Dict = defaultdict()
Positive_Word_Dict = defaultdict()
Sentence_Sentiment = {}
Evaluation_Dict = OrderedDict()

frequent_word_list = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
ID_Val = []
#Basic data cleaning using regular expressions. I have removed the frequently repeated words
#as described in the above list. I have eliminated all kind of punctuation marks - like .,!,-, etc
for line in open("hotelNegT-train.txt",encoding="utf8").readlines():
    if line.strip():
        word_split = line.replace(',',' ').replace('.',' ').replace('!',' ').replace('--',' ').split()
        for each_word in word_split:
                        if re.match(r'[IDidIdiD].*-[0-9].*',each_word):
                            count_negative_ID+=1
                            ID_Val.append(each_word)
                            continue
                        elif re.match(r'(\d+)\.(\d+)+',each_word):
                            continue
                        elif re.match(r'(\d+)+',each_word):
                            continue 
                        elif re.match(r'$(\d+)+',each_word):
                            continue  
                        elif re.match(r'[A-Za-z]*\.+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'[A-Za-z]*\?+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'[A-Za-z]*\!+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'-+',each_word):
                            continue
                        elif re.match(r'[A-Za-z]*\)+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'\(+[A-Za-z]*',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'[A-Za-z]*\,+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                            
                        if each_word in frequent_word_list:
                            continue
                        else:
                            list_negative_word.append(each_word)

List_Size_Negative = len(list_negative_word)
Negative_Word_Dict =  Counter(list_negative_word)
Denominator_Sum_Neg = sum(Negative_Word_Dict.values())
Unique_list_negative_word = set(list_negative_word)
Unique_List_Size_Negative = len(Unique_list_negative_word)
print (ID_Val)
list_positive_word = []
for line in open("hotelPosT-train.txt",encoding="utf8").readlines():
    if line.strip():
        word_split = line.split()
        word_split = line.replace(',',' ').replace('.',' ').replace('!',' ').replace('--',' ').split()
        for each_word in word_split:
                        if re.match(r'ID-[0-9].*',each_word):
                            count_positive_ID+=1
                            ID_Val = each_word
                            continue
                        elif re.match(r'(\d+)\.(\d+)+',each_word):
                            continue
                        elif re.match(r'(\d+)+',each_word):
                            continue 
                        elif re.match(r'$(\d+)+',each_word):
                            continue  
                        elif re.match(r'[A-Za-z]*\.+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'[A-Za-z]*\?+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'[A-Za-z]*\!+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'-+',each_word):
                            continue
                        elif re.match(r'[A-Za-z]*\)+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'\(+[A-Za-z]*',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'[A-Za-z]*\,+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'[A-Za-z]*\/+[A-Za-z]*',each_word):
                            list_word = each_word.split('/')
                            if list_word[0] not in frequent_word_list:
                                list_negative_word.append(list_word [0])
                            else:
                                continue
                            
                            if list_word[1] not in frequent_word_list:
                                list_negative_word.append(list_word [1])
                            else:
                                continue
                        if each_word in frequent_word_list:
                            continue
                        else:
                            list_positive_word.append(each_word)
#List of all words in positive training set with repitition
List_Size_Positive = len(list_positive_word)
#Dictionary having the words : count 
Positive_Word_Dict = Counter(list_positive_word)
Denominator_Sum_Pos = sum(Positive_Word_Dict.values())

#List of all unique words in Positive Training set
Unique_list_positive_word = set(list_positive_word)
#Length of Unique Size list
Unique_List_Size_Positive = len(Unique_list_positive_word)
Total_No_of_Docs = count_positive_ID + count_negative_ID
print ("count_positive_ID - ",count_positive_ID)
print ("count_negative_ID - ",count_negative_ID)
print ("Total_No_of_Docs - ",Total_No_of_Docs)

Log_Prior_Positive = np.log10(float(count_positive_ID)/float(Total_No_of_Docs))
Log_Prior_Negative = np.log10(float(count_negative_ID)/float(Total_No_of_Docs))
print ("Log_Prior_Positive - ",Log_Prior_Positive)
print ("Log_Prior_Negative - ",Log_Prior_Negative)

Negative_Word_Dict_final = defaultdict()
Positive_Word_Dict_final = defaultdict()

print (Denominator_Sum_Pos)
print (Denominator_Sum_Neg)
#Creating maximum likelihood estimate values for each word in positive training set
for key in Positive_Word_Dict:    
    val = np.log10 (float(Positive_Word_Dict[key] + 1)/float(Denominator_Sum_Pos + Unique_List_Size_Positive))
    Positive_Word_Dict_final.update({key:val})
for key in Negative_Word_Dict:    
    val =  np.log10(float(Negative_Word_Dict[key] + 1)/float(Denominator_Sum_Neg + Unique_List_Size_Negative))
    Negative_Word_Dict_final.update({key:val})

Negative_Probability_Sentence = 0.00
Positive_Probability_Sentence = 0.00
Sentence_Token_List = []
Sentence_List = []

with open("TestSet.txt",'r',encoding="utf8") as f:
   token =  [line.split() for line in f]
   for each_word in token:
        Sentence_Token_List.append(each_word)

Test_Review_Dict = OrderedDict()
Test_Review_Class = OrderedDict()
temp_dict = {}
List_Tokens_ID = []
temp = []

for i in range (0,len(Sentence_Token_List)):
    for j in range (0,len(Sentence_Token_List[i])):
        word = Sentence_Token_List[i][j]
        if re.match(r'ID-[0-9].*',word):
            Id_Value = word
            continue
        elif re.match(r'(\d+)\.(\d+)+',word):
            continue
        elif re.match(r'(\d+)+',word):
            continue 
        elif re.match(r'$(\d+)+',word):
            continue  
        elif re.match(r'[A-Za-z]*\.+',word):
            word = re.sub(r'[^\w\s]','',word)
        elif re.match(r'[A-Za-z]*\?+',word):
            word = re.sub(r'[^\w\s]','',word)
        elif re.match(r'[A-Za-z]*\!+',word):
            word = re.sub(r'[^\w\s]','',word)
        elif re.match(r'-+',word):
            continue
        elif re.match(r'[A-Za-z]*\)+',word):
            word = re.sub(r'[^\w\s]','',word)
        elif re.match(r'\(+[A-Za-z]*',word):
            word = re.sub(r'[^\w\s]','',word)
        elif re.match(r'[A-Za-z]*\,+',word):
            word = re.sub(r'[^\w\s]','',word)
        elif re.match(r'[A-Za-z]*\/+[A-Za-z]*',word):
            list_word = word.split('/')
            if list_word[0] in frequent_word_list:
                continue
            else:
                List_Tokens_ID.append(list_word[0])
            if list_word[1] in frequent_word_list:
                continue
            else:
                List_Tokens_ID.append(list_word[0])
                
        if (word not in frequent_word_list):
                List_Tokens_ID.append(word)
    temp = copy.copy(List_Tokens_ID)
    Test_Review_Dict.update ({Id_Value:temp})
    List_Tokens_ID[:] = []

for each_key,each_value in Test_Review_Dict.items():
    ID_Value = each_key
    for val in each_value:
        if ((val in Negative_Word_Dict_final) and (val in Positive_Word_Dict_final)):
            Negative_Probability_Sentence += Negative_Word_Dict_final[val]
            Positive_Probability_Sentence += Positive_Word_Dict_final[val]

    Positive_Probability_Sentence+=Log_Prior_Positive
    Negative_Probability_Sentence+=Log_Prior_Negative
    temp_dict.update({'POS': Positive_Probability_Sentence})
    temp_dict.update({'NEG': Negative_Probability_Sentence})
    print ("ID- Value : {}, Negative : {} , Positive : {}".format(ID_Value, Negative_Probability_Sentence,Positive_Probability_Sentence))
    key = [k for k,v in temp_dict.items() if v==max(temp_dict.values())][0] 
    Test_Review_Class.update ({ID_Value:key})
    temp_dict.clear()
    Positive_Probability_Sentence = 0.0
    Negative_Probability_Sentence = 0.0
    
with open("Upadhyaya-Swathi-assgn3-out_BasicNaiveBayes.txt",'w') as ofile:
        for keys,values in Test_Review_Class.items():
            ofile.write((str(keys) + '\t' + values + '\n'))

####with open("Evaluation.txt",'w') as ofile:
####        for keys,values in Evaluation_Dict.items():
####            ofile.write((str(keys) + '\t' + values + '\n'))
##wrong_ans = 0
##for line in open('Evaluation.txt','r').readlines():
##	if line.strip():
##	    word_split = line.split()
##	    Evaluation_Dict.update({word_split[0] : word_split[1]})
##			
##for key in Test_Review_Class.keys():
##    if key in Evaluation_Dict.keys():
##        count+=1
##        if Evaluation_Dict[key] == Test_Review_Class[key]:
##            accuracy+=1
##        else:
##            #print ("TAG DONT MATCH!")
##            wrong_ans+=1
##    else:
##        print ("Key NOT Found!")
##print (float(accuracy)/float(count))
##print ("Wrong Answers - ",wrong_ans)

def NaiveBayes_BuiltIn_Package():
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

    with open("Output.txt",'w') as ofile:
            for key,val in dictionary.items():
                ofile.write((str(key) + '\t' + str(val) + '\n'))

NaiveBayes_BuiltIn_Package()
