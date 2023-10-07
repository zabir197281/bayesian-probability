import numpy as np
import matplotlib.pyplot as plt
import random


# Making random classifier that assigns a random class to each test semples and computing the accuricy  

test_male_female_class_file="male_female_y_test.txt"
with open(test_male_female_class_file,'r',encoding="utf8") as f:
    test_male_female_class = [x for x in f.readlines()]

np_test_male_female_class=np.array(test_male_female_class)
np_test_male_female_class_float = np.asarray(np_test_male_female_class, dtype=float)

# Assigning rendom class.

random_class_list = []

for  i in range(len(np_test_male_female_class_float)):
    random_value = random.randint(0, 1)
    random_class_list.append(random_value)

np_random_class_list=np.array(random_class_list)

np_random_class_list_float = np.asarray(np_random_class_list, dtype=float)

# Classes of test data

test_labels =np_test_male_female_class_float

# Random calss

random_predicted_labels=np_random_class_list_float



random_correct_predictions=0

for test_label, random_predicted_label in zip(test_labels, random_predicted_labels):
    if test_label == random_predicted_label:
        random_correct_predictions += 1

number_of_test_semples= len(test_labels)

# Calculating accuricy for random classes

accuracy_random = random_correct_predictions / number_of_test_semples

print()

print(f" Accuricy for Random Classifier : {round(accuracy_random*100,3)} %")

# Making most likely classifier that assigns the most likely class to each test semples and computing the accuricy  

# Exerting the data from "male_female_y_train.txt"

male_female_class_file="male_female_y_train.txt"
with open(male_female_class_file,'r',encoding="utf8") as f:
    male_female_class = [x for x in f.readlines()]

np_male_female_class=np.array(male_female_class)
np_male_female_class_float = np.asarray(np_male_female_class, dtype=float)

male_mask= (np_male_female_class_float==0)

number_of_male=sum(male_mask)

female_mask= (np_male_female_class_float==1)

number_of_female=sum(female_mask)

# Assigning the most likely class to test sempel

assigned_test_class=[]

if(number_of_male<number_of_female):
    assigned_test_class=[1]*len(np_test_male_female_class_float)
    
else:
    assigned_test_class=[0]*len(np_test_male_female_class_float)

np_assigned_test_class=np.array(assigned_test_class)
np_assigned_test_class_float=np.asarray(np_assigned_test_class, dtype=float)
assigned_predicted_levels = np_assigned_test_class_float

# Calculating accuricy for most likely classes

likelihood_correct_predictions=0

for test_label, assigned_predicted_level in zip(test_labels, assigned_predicted_levels):
    if test_label == assigned_predicted_level:
        likelihood_correct_predictions += 1

accuracy_likelihood= likelihood_correct_predictions / number_of_test_semples

print()

print(f" Accuricy for Most Likely Classifier : {round(accuracy_likelihood*100,3)} %")

print()