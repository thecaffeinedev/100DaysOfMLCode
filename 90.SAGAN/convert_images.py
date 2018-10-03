import tensorflow as tf
import numpy as np
from PIL import Image
import os


IN_PATH = './data/train'
OUT_PATH = './data/'


class TFRecordExporter(tf.python_io.TFRecordWriter):
    def __init__(self, path):
        super().__init__(path)

    def add_image(self, img):
        features = tf.train.Features(feature={
            'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=img.shape)),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()]))})
        ex = tf.train.Example(features=features)
        self.write(ex.SerializeToString())


def convert(in_path, out_path, resolution):
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    file_list = os.listdir(in_path)
    file_list = [os.path.join(in_path, f) for f in file_list]

    exporter = TFRecordExporter(os.path.join(out_path, 'data.tfrecord'))
    tf.logging.set_verbosity(tf.logging.INFO)

    for i, f in enumerate(file_list):
        img = Image.open(f)

        aspect = img.size[0] / img.size[1]
        if aspect > 1:
            width = int(resolution * aspect)
            offset = (width - resolution) // 2
            img = img.resize((width, resolution))
            img = img.crop((offset, 0, offset + resolution, resolution))
        else:
            height = int(resolution / aspect)
            offset = (height - resolution) // 2
            img = img.resize((resolution, height))
            img = img.crop((0, offset, resolution, offset + resolution))

        exporter.add_image(np.asarray(img))
        tf.logging.log_every_n(tf.logging.INFO,
                               'Write image %d/%d' % (i, len(file_list)),
                               100)


if __name__ == '__main__':
    convert(IN_PATH, OUT_PATH, 64)
