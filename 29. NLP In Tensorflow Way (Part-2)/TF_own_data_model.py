'''
positive negative sentiment analysis 
applying TF to a realist data set 
# to try to buld a classifier classifying positive and negative sentiments 
'''

# [chair, table, spoon, television ] - Lexicon created
# entire data set has only these unique words so this
# is our global lexicon

# now we have anew sentence coming chair up to the table
# this has to be converted
'''
I pulled the chair up to the table 
Step 1: Lexicon will be:
            np.zeroes(len(lexicon))
            [0 0 0 0] 
            I - no Pulled - No The - No
            Chair - Yes; index of chair is 1 
            So [1 0 0 0] 
            to - no the - no Table - 1 
            So [1 1 0 0] - thus this is our vector for this specific sentence
            we'll have to do this for all the sentences across our dictionary 
'''
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

# concept of tokenizing; lemmatizing; stemming

'''
tokenizing - takes the words from the statement and separates them 
    $ i pulled the chair up to the table
    # [i, pulled, the, chair] - word tokenize  
Lemmatizing - sentiment/reasons/word meaning/reasons
            - running/ran - RUN 
            ex: I like this product
                I used to like this product
                        (Tense does matter but 
'''
lemmatizer = WordNetLemmatizer()
hm_lines = 10000000


# memoryError - ran out of Ram
# COULD use less nodes/layers in NN to avoid Memory Error
# but accuracy might go down
# lexicon - find all words in pos and neg data sets
def create_lexicon(pos, neg):
    lexicon = []
    for file in [pos, neg]:
        with open(file, 'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:  # upto however many lines we're gonna read
                all_words = word_tokenize(l.lower())  # tokenizing words per line
                lexicon += list(all_words)

    # firs thing - we're gonna lemmatize all these words
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    # stemming them into legitimate words
    # Input Vector is going to be that input vector
    # ieally - LEXICON should be shortest possible so that
    # we can have a decent sized model in terms of 3 layers
    # 1000
    w_counts = Counter(lexicon)
    # this gives us a dictionary like elements
    # w_counts = {'the':52000, 'and',:22323} EXAMPLE
    l2 = []
    for w in w_counts:
        if 1000 > w_counts[w] > 50:
            l2.append(w)
            # because we dont want super common words like 'the' 'and' 'or' etc. - NOT VALUABLE
    print(len(l2))
    return l2
    # l2 is the final lexicon


def sample_handling(sample, lexicon, classification):
    featureset = []  # [1 0] pos sentiment [0 1] negative sentiment
    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            #print(features)
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    # like the example discussed earlier
                    features[index_value] += 1
            features = list(features)
            featureset.append([features, classification])
            #print(featureset)

    return featureset


def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling('pos.txt', lexicon, [1, 0])

    features += sample_handling('neg.txt', lexicon, [0, 1])
    random.shuffle(features)
    print(features)
    # does tf.agrmax([output]) == tf.argmax)[expectations]) was  the final question
    # this was the question earlier

    # want to shuffle bcause that's how NN Model works - it's going to be shifting the
    # weights for RNN model to work
    features = np.array(features)
    testing_size = int(test_size * len(features))

    # [[5, 8], [7,9]] want all 0th elements
    # [:,0] implies you want all of the 0th elements
    # [5,7] is what you'll get

    # [[[0 1 1 0 1], [0,1]],
    # [features, label ],
    # [features, label]] # features themselves are little one hot arrays

    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])

    test_x = list(features[:, 0][:-testing_size:])
    test_y = list(features[:, 1][:-testing_size:])

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)