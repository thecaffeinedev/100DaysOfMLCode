import tensorflow as tf
import numpy as np
from collections import defaultdict
from rbm import RBM
from dataset import load_dataset
from utils import chunker, revert_expected_value, expand, iteration_str
import sys
from math import sqrt

tf.flags.DEFINE_integer("epochs", 100, "")
tf.flags.DEFINE_integer("batch_size", 10, "")
tf.flags.DEFINE_integer("num_hidden", 100, "")
tf.flags.DEFINE_float("decay", 0.01, "")
tf.flags.DEFINE_float("momentum", 0.9, "")
tf.flags.DEFINE_float("l_v", 0.01, "")
tf.flags.DEFINE_float("l_w", 0.01, "")
tf.flags.DEFINE_float("l_h", 0.01, "")
tf.flags.DEFINE_string("train_path", "ml-100k/u1.base", "")
tf.flags.DEFINE_string("test_path", "ml-100k/u1.test", "")
tf.flags.DEFINE_string("sep", "\t", "")
FLAGS = tf.flags.FLAGS


if __name__ == "__main__":
    all_users, all_movies, tests = load_dataset(FLAGS.train_path, FLAGS.test_path,
                                                FLAGS.sep, user_based=True)
    rbm = RBM(len(all_movies) * 5, FLAGS.num_hidden)
    print("model created")
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    profiles = defaultdict(list)
    with open(FLAGS.train_path, 'rt') as data:
        for i, line in enumerate(data):
            uid, mid, rat, timstamp = line.strip().split(FLAGS.sep)
            profiles[uid].append((mid, float(rat)))
    print("Users and ratings loaded")
    for e in range(FLAGS.epochs):
        
        for batch_i, batch in enumerate(chunker(list(profiles.keys()),
                                                FLAGS.batch_size)):
            size = min(len(batch), FLAGS.batch_size)
            
            # create needed binary vectors
            bin_profiles = {}
            masks = {}
            for userid in batch:
                user_profile = np.array([0.] * len(all_movies))
                mask = [0] * (len(all_movies) * 5)
                for movie_id, rat in profiles[userid]:
                    user_profile[all_movies.index(movie_id)] = rat
                    for _i in range(5):
                        mask[5 * all_movies.index(movie_id) + _i] = 1
                example = expand(np.array([user_profile])).astype('float32')
                bin_profiles[userid] = example
                masks[userid] = mask

            profile_batch = [bin_profiles[el] for el in batch]
            masks_batch = [masks[id] for id in batch]
            train_batch = np.array(profile_batch).reshape(size,
                                                          len(all_movies * 5))
            train_masks = np.array(masks_batch).reshape(size,
                                                        len(all_movies) * 5)
            _  = sess.run([rbm.optimizer], feed_dict={rbm.input: train_batch, rbm.mask : masks_batch})
            sys.stdout.write('.')
            sys.stdout.flush()
        
        # test step
        ratings = []
        predictions = []
        for batch in chunker(list(tests.keys()), FLAGS.batch_size):
            size = min(len(batch), FLAGS.batch_size)

            # create needed binary vectors
            bin_profiles = {}
            masks = {}
            for userid in batch:
                user_profile = [0.] * len(all_movies)
                mask = [0] * (len(all_movies) * 5)
                for movie_id, rat in profiles[userid]:
                    user_profile[all_movies.index(movie_id)] = rat
                    for _i in range(5):
                        mask[5 * all_movies.index(movie_id) + _i] = 1
                example = expand(np.array([user_profile])).astype('float32')
                bin_profiles[userid] = example
                masks[userid] = mask

            positions = {profile_id: pos for pos, profile_id
                         in enumerate(batch)}
            profile_batch = [bin_profiles[el] for el in batch]
            test_batch = np.array(profile_batch).reshape(size,
                                                         len(all_movies * 5))
            predict = sess.run(rbm.predict, feed_dict={rbm.input : test_batch})
            user_preds = revert_expected_value(predict)
 
            for profile_id in batch:
                test_movies = tests[profile_id]
                try:
                    for movie, rating in test_movies:
                        current_profile = user_preds[positions[profile_id]]
                        predicted = current_profile[all_movies.index(movie)]
                        rating = float(rating)
                        ratings.append(rating)
                        predictions.append(predicted)
                except Exception:
                    pass

        vabs = np.vectorize(abs)
        distances = np.array(ratings) - np.array(predictions)

        mae = vabs(distances).mean()
        rmse = sqrt((distances ** 2).mean())
        print("\nepoch: {}, mae/rmse: {}/{}".format(e, mae, rmse))
