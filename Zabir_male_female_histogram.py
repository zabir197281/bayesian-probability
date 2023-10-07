import numpy as np
import matplotlib.pyplot as plt
import random

# extracting the hight and weight from the train data.

Hight_weight_file="male_female_X_train.txt"
with open(Hight_weight_file,'r',encoding="utf8") as f:
    height_weight = [x.rstrip().split(' ') for x in f.readlines()]

np_height_weight = np.array(height_weight)

np_height_weight_float = np.asarray(np_height_weight, dtype=float)

# extracting the classes of male and female from the train data where male is "0" and female is "1".

male_female_class_file="male_female_y_train.txt"
with open(male_female_class_file,'r',encoding="utf8") as f:
    male_female_class = [x for x in f.readlines()]

np_male_female_class=np.array(male_female_class)
np_male_female_class_float = np.asarray(np_male_female_class, dtype=float)

# storing the male height in the male_heights valiable
male_heights = np_height_weight_float[np_male_female_class_float == 0][:, 0]

# storing the female height in the female_heights valiable
female_heights = np_height_weight_float[np_male_female_class_float == 1][:, 0]

# storing the male weight in the male_weight valiable
male_weight = np_height_weight_float[np_male_female_class_float == 0][:, 1]

# storing the female weight in the female_weight valiable
female_weight = np_height_weight_float[np_male_female_class_float == 1][:, 1]

height_bin_range = [80, 220]
weight_bin_range = [30, 180]


print("After estimating visially height and weight histogram for both the classes, we can say")
print("that height is more better for classification because for height the overlaping for male")
print("and female class as much lesser the weight.")

# Histogram figure for height for both the classes 

plt.figure()
plt.hist(male_heights, bins=10,range=height_bin_range, alpha=0.5, label='Male', color='blue',edgecolor='black')
plt.hist(female_heights, bins=10,range=height_bin_range, alpha=0.5, label='Female', color='pink',edgecolor='black')
plt.grid(color='gray')
plt.xlabel('Height')
plt.ylabel('Frequency')
plt.legend()


# Histogram figure for weight for both the classes 

plt.figure()
plt.hist(male_weight, bins=10,range=weight_bin_range, alpha=0.5, label='Male', color='blue',edgecolor='black')
plt.hist(female_weight, bins=10,range=weight_bin_range, alpha=0.5, label='Female', color='pink',edgecolor='black')
plt.grid(color='gray')
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.legend()
plt.show()

"""
After estimating visially height and weight histogram for both the classes, we can say
that height is more better for classification because for height the overlaping for male
and female class as much lesser the weight.

"""

