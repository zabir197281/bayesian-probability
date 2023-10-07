import numpy as np
import matplotlib.pyplot as plt

# Extracting height and weight from "male_female_X_train.txt"

Hight_weight_file="male_female_X_train.txt"
with open(Hight_weight_file,'r',encoding="utf8") as f:
    height_weight = [x.rstrip().split(' ') for x in f.readlines()]

np_height_weight = np.array(height_weight)

np_height_weight_float = np.asarray(np_height_weight, dtype=float)

# Extracting male and female class from "male_female_y_train.txt" where male is 0 and female in 1

male_female_class_file="male_female_y_train.txt"
with open(male_female_class_file,'r',encoding="utf8") as f:
    male_female_class = [x for x in f.readlines()]

np_male_female_class=np.array(male_female_class)
np_male_female_class_float = np.asarray(np_male_female_class, dtype=float)

male_heights = np_height_weight_float[np_male_female_class_float == 0][:, 0]
female_heights = np_height_weight_float[np_male_female_class_float == 1][:, 0]

male_weight = np_height_weight_float[np_male_female_class_float == 0][:, 1]
female_weight = np_height_weight_float[np_male_female_class_float == 1][:, 1]

# Computing the probability of male

mask_male=(np_male_female_class_float==0)
number_of_male=sum(mask_male)

priobability_of_male= number_of_male/len(np_male_female_class_float)

# Computing the probability of female

mask_female=(np_male_female_class_float==1)
number_of_female=sum(mask_female)

priobability_of_female= number_of_female/len(np_male_female_class_float)


height_bin_range = [80, 220]
weight_bin_range = [30, 180]

# computing the probability of height

hist_male_h, bins_male_h = np.histogram(male_heights, bins=10, range=height_bin_range)
hist_female_h, bins_female_h = np.histogram(female_heights, bins=10, range=height_bin_range)

# Finding the centers of each bins

male_female_bin_centers_h = (bins_male_h[:-1] + bins_male_h[1:]) / 2

# function computing the probability of height

def compute_probability_of_height(h):
    subtracted_bin_centers=np.abs(male_female_bin_centers_h-h) 
    h_bin_index=np.argsort(subtracted_bin_centers)[0]
    h_bin_index_male_female=hist_male_h[h_bin_index]+hist_female_h[h_bin_index]
    probability_of_h=h_bin_index_male_female/len(np_male_female_class_float)
    return probability_of_h

# function Computing the Probability of Height Given male

def compute_probability_of_height_given_male(h):
    subtracted_bin_centers=np.abs(male_female_bin_centers_h-h)
    h_bin_index=np.argsort(subtracted_bin_centers)[0]
    h_bin_index_male=hist_male_h[h_bin_index]
    probability_of_h_male=h_bin_index_male/number_of_male
    return probability_of_h_male

# function Computing the Probability of Height Given female

def compute_probability_of_height_given_female(h):
    subtracted_bin_centers=np.abs(male_female_bin_centers_h-h)
    h_bin_index=np.argsort(subtracted_bin_centers)[0]
    h_bin_index_female=hist_female_h[h_bin_index]
    probability_of_h_female=h_bin_index_female/number_of_female
    return probability_of_h_female

#Compution probability of weight

hist_male_w, bins_male_w = np.histogram(male_weight, bins=10, range=weight_bin_range)
hist_female_w, bins_female_w = np.histogram(female_weight, bins=10, range=weight_bin_range)

# Computing the centers of each bins

male_female_bin_centers_w = (bins_male_w[:-1] + bins_male_w[1:]) / 2

# Function to compute the probability of weight

def compute_probability_of_weight(w):
    subtracted_bin_centers=np.abs(male_female_bin_centers_w-w)
    w_bin_index=np.argsort(subtracted_bin_centers)[0]
    w_bin_index_male_female=hist_male_w[w_bin_index]+hist_female_w[w_bin_index]
    probability_of_w=w_bin_index_male_female/len(np_male_female_class_float)
    return probability_of_w

# Function to compute Probability of weight given male

def compute_probability_of_weight_given_male(w):
    subtracted_bin_centers=np.abs(male_female_bin_centers_w-w)
    w_bin_index=np.argsort(subtracted_bin_centers)[0]
    w_bin_index_male=hist_male_w[w_bin_index]
    probability_of_w_male=w_bin_index_male/number_of_male
    return probability_of_w_male


# Function to compute Probability of weight given female

def compute_probability_of_weight_given_female(w):
    subtracted_bin_centers=np.abs(male_female_bin_centers_w-w)
    w_bin_index=np.argsort(subtracted_bin_centers)[0]
    w_bin_index_female=hist_female_w[w_bin_index]
    probability_of_w_female=w_bin_index_female/number_of_female
    return probability_of_w_female

# Function to compute the probability of male of female given the height

def probability_of_male_or_female_given_the_hight(h):
    probability_of_male_given_h=(compute_probability_of_height_given_male(h)*priobability_of_male)/compute_probability_of_height(h)
    probability_of_female_given_h=(compute_probability_of_height_given_female(h)*priobability_of_female)/compute_probability_of_height(h)
    
    if probability_of_female_given_h > probability_of_male_given_h:
        return 1
    else:
        return 0
    
# Function to compute the probability of male of female given the Weight

def probability_of_male_or_female_given_the_weight(w):
    probability_of_male_given_w=(compute_probability_of_weight_given_male(w)*priobability_of_male)/compute_probability_of_weight(w)
    probability_of_female_given_w=(compute_probability_of_weight_given_female(w)*priobability_of_female)/compute_probability_of_weight(w)
    
    if probability_of_female_given_w > probability_of_male_given_w:
        return 1
    else:
        return 0
    

# Extracting height and weight from "male_female_X_test.txt"


test_hight_weight_file="male_female_X_test.txt"
with open(test_hight_weight_file,'r',encoding="utf8") as f:
    height_weight_test = [x.rstrip().split(' ') for x in f.readlines()]

np_height_weight_test = np.array(height_weight_test)

np_height_weight_test_float = np.asarray(np_height_weight_test, dtype=float)

# Extracting male and female class from "male_female_y_test.txt"

test_male_female_class_file="male_female_y_test.txt"
with open(test_male_female_class_file,'r',encoding="utf8") as f:
    test_male_female_class = [x for x in f.readlines()]

np_test_male_female_class=np.array(test_male_female_class)
np_test_male_female_class_float = np.asarray(np_test_male_female_class, dtype=float)


np_test_height=np_height_weight_test_float[:,0]

np_test_weight=np_height_weight_test_float[:,1]


# calculating male or female given the height

predicted_class_given_height= [ probability_of_male_or_female_given_the_hight(i) for i in np_test_height]

# calculating the accuricy for height

correct_predictions_with_height=0

for test_label, predicted_label in zip(np_test_male_female_class_float, predicted_class_given_height):
    if test_label == predicted_label:
        correct_predictions_with_height += 1

accuracy_with_height = correct_predictions_with_height / len(np_test_male_female_class_float)

print()

print(f"Accuracy for height only : {round(accuracy_with_height*100,3)} %")


# calculating male or female given the weight

predicted_class_given_weight= [ probability_of_male_or_female_given_the_weight(i) for i in np_test_weight]

# calculating the accuricy for weight


correct_predictions_with_weight=0

for test_label, predicted_label in zip(np_test_male_female_class_float, predicted_class_given_weight):
    if test_label == predicted_label:
        correct_predictions_with_weight += 1

accuracy_with_weight = correct_predictions_with_weight / len(np_test_male_female_class_float)

print()

print(f"Accuracy for weight only : {round(accuracy_with_weight*100,3)} %")


# Function to compute the probability of male of female given the height and weight togather


def probability_of_male_or_female_given_the_height_and_weight(h,w):

    probability_of_male_given_h=(compute_probability_of_height_given_male(h)*priobability_of_male)/compute_probability_of_height(h)
    probability_of_female_given_h=(compute_probability_of_height_given_female(h)*priobability_of_female)/compute_probability_of_height(h)


    probability_of_male_given_w=(compute_probability_of_weight_given_male(w)*priobability_of_male)/compute_probability_of_weight(w)
    probability_of_female_given_w=(compute_probability_of_weight_given_female(w)*priobability_of_female)/compute_probability_of_weight(w)
    
    probability_of_male_given_h_and_w=probability_of_male_given_h*probability_of_male_given_w
    probability_of_female_given_h_and_w=probability_of_female_given_h*probability_of_female_given_w

    if probability_of_female_given_h_and_w > probability_of_male_given_h_and_w:
        return 1
    else:
        return 0


predicted_class_given_Height_and_weight= [ probability_of_male_or_female_given_the_height_and_weight(i[0],i[1]) for i in np_height_weight_test_float]



# calculating the accuricy for height and  weight


correct_predictions_with_height_and_weight=0

for test_label, predicted_label in zip(np_test_male_female_class_float, predicted_class_given_Height_and_weight):
    if test_label == predicted_label:
        correct_predictions_with_height_and_weight += 1

accuracy_with_height_weight = correct_predictions_with_height_and_weight / len(np_test_male_female_class_float)

print()

print(f"Accuracy for weight and height together : {round(accuracy_with_height_weight*100,3)} %")

print()