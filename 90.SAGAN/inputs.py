import tensorflow as tf
import os


class ImageInputs:
    def __init__(self, data_path, batch_size, res, repeat=True, shuffle=True, augment=True):
        self.batch_size = batch_size
        self.res = res
        self.shuffle = shuffle

        dataset = tf.data.TFRecordDataset(data_path)
        dataset = dataset.map(self._parse)
        if shuffle:
            dataset = dataset.shuffle(500)
        if repeat:
            dataset = dataset.repeat()
        if augment:
            dataset = dataset.map(self._augment)

        self.dataset = dataset.batch(batch_size)

    def _parse(self, serialized):
        features = {'image': tf.FixedLenFeature([], tf.string),
                    'shape': tf.FixedLenFeature([3], tf.int64)}

        example = tf.parse_single_example(serialized, features)

        image = tf.decode_raw(example['image'], tf.uint8)
        image = tf.reshape(image, example['shape'])

        image = tf.image.per_image_standardization(tf.to_float(image))

        return image

    def _augment(self, image):
        image = tf.image.random_flip_left_right(image)
        angle = tf.random_normal([], stddev=5.0 / 180.0)
        image = tf.contrib.image.rotate(image, angle)

        return image

    def get_next(self, name='batch'):
        images = self.dataset.make_one_shot_iterator().get_next(name)
        images.set_shape([self.batch_size, self.res, self.res, 3])

        return images
