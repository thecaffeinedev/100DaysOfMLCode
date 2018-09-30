#Program on collabortive technique 

from dataset import dataset
from math import sqrt

print('**************Dhananjaya Panda and Vikas Singh ratings**************')
print(dataset['Dhananjaya Panda'])
print(dataset['Vikash Singh'])
def similarity_score(person1,person2):
	
	# Returns ratio Euclidean distance score of person1 and person2 
 
	both_viewed = {}		# To get both rated items by person1 and person2
 
	for item in dataset[person1]:
		if item in dataset[person2]:
			both_viewed[item] = 1
 
		# Conditions to check they both have an common rating items	
		if len(both_viewed) == 0:
			return 0
 
		# Finding Euclidean distance 
		sum_of_eclidean_distance = []	
 
		for item in dataset[person1]:
			if item in dataset[person2]:
				sum_of_eclidean_distance.append(pow(dataset[person1][item] - dataset[person2][item],2))
		sum_of_eclidean_distance = sum(sum_of_eclidean_distance)
 
		return 1/(1+sqrt(sum_of_eclidean_distance))
print("predicted similarity score using ucledean distance algorithm:")
print similarity_score('Dhananjaya Panda','Vikash Singh')
def similarity_score(person1,person2):
	
	# Returns ratio Euclidean distance score of person1 and person2 
 
	both_viewed = {}		# To get both rated items by person1 and person2
 
	for item in dataset[person1]:
		if item in dataset[person2]:
			both_viewed[item] = 1
 
		# Conditions to check they both have an common rating items	
		if len(both_viewed) == 0:
			return 0
 
		# Finding Euclidean distance 
		sum_of_eclidean_distance = []	
 
		for item in dataset[person1]:
			if item in dataset[person2]:
				sum_of_eclidean_distance.append(pow(dataset[person1][item] - dataset[person2][item],2))
		sum_of_eclidean_distance = sum(sum_of_eclidean_distance)
 
		return 1/(1+sqrt(sum_of_eclidean_distance))
 
def pearson_correlation(person1,person2):
 
	# To get both rated items
	both_rated = {}
	for item in dataset[person1]:
		if item in dataset[person2]:
			both_rated[item] = 1
 
	number_of_ratings = len(both_rated)		
	
	# Checking for number of ratings in common
	if number_of_ratings == 0:
		return 0
 
	# Add up all the preferences of each user
	person1_preferences_sum = sum([dataset[person1][item] for item in both_rated])
	person2_preferences_sum = sum([dataset[person2][item] for item in both_rated])
 
	# Sum up the squares of preferences of each user
	person1_square_preferences_sum = sum([pow(dataset[person1][item],2) for item in both_rated])
	person2_square_preferences_sum = sum([pow(dataset[person2][item],2) for item in both_rated])
 
	# Sum up the product value of both preferences for each item
	product_sum_of_both_users = sum([dataset[person1][item] * dataset[person2][item] for item in both_rated])
 
	# Calculate the pearson score
	numerator_value = product_sum_of_both_users - (person1_preferences_sum*person2_preferences_sum/number_of_ratings)
	denominator_value = sqrt((person1_square_preferences_sum - pow(person1_preferences_sum,2)/number_of_ratings) * (person2_square_preferences_sum -pow(person2_preferences_sum,2)/number_of_ratings))
	if denominator_value == 0:
		return 0
	else:
		r = numerator_value/denominator_value
		return r

print("similarity score using pearson correlation technique:") 
print pearson_correlation('Dhananjaya Panda','Vikash Singh')
