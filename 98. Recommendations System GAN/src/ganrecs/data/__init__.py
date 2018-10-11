#!/usr/bin/env python3
from random import sample
from random import shuffle


class Rating():

    def __init__(self, data_tuple):
        self.user, self.item, self.rating, _ = data_tuple
    
    def as_list(self):
        return [self.user, self.item, self.rating]


class RatingCollection():

    SAMPLE_AMOUNT = 0.2
    FOLDS = 10

    def __init__(self, data_collection):
        sample_size = int(len(data_collection) * self.SAMPLE_AMOUNT)
        # _sample = sample(data_collection, sample_size)
        # data_collection = [x for x in data_collection if x not in _sample]
        _sample = []
        self.dropouts = [Rating(i) for i in _sample]
        self.ratings = [Rating(i) for i in data_collection]
        self._get_cv()

    def all_ratings(self):
        return self.dropouts + self.ratings

    def _get_cv(self):
        _size = len(self.ratings)
        shuffle(self.ratings)
        _cv_size = int(_size / self.FOLDS)
        self.folds = []
        marker = 0
        for i in range(self.FOLDS):
            self.folds.append(self.ratings[marker:marker+_cv_size])
            marker = marker + _cv_size
        if _size * self.FOLDS < len(self.ratings):
            self.folds[-1] = self.folds[-1] + self.ratings[-1:-1 - (len(self.ratings) - _cv_size * self.FOLDS)]

    def _get_matrix(self, ratings):
        user_tuples = {}
        movies = set([r.item for r in self.ratings])
        for rating in ratings:
            if rating.user not in user_tuples.keys():
                user_tuples[rating.user] = {int(r):0 for r in movies}
            user_tuples[rating.user][int(rating.item)] = float(rating.rating) / 5.
        return user_tuples

    def __iter__(self):
        for fold in self.folds:
            yield self._get_matrix(fold)
