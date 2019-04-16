import tensorflow as tf
import pathlib
import random


class ImageIterator(object):
    def __init__(self, data_root, batch_size, image_size, image_channels):
        self.data_root = pathlib.Path(data_root)
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_channels = image_channels


    def preprocess_image(self, image, isCrop = False):
        image = tf.image.decode_jpeg(image, channels=self.image_channels)
        print(image.get_shape().as_list())
        if isCrop:
            face_width = face_height = 128
            offset_height = (tf.shape(image)[0]-face_height)//2
            offset_width = (tf.shape(image)[1]-face_width)//2
            image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, face_height, face_width)


        image = tf.image.resize_images(image, [self.image_size, self.image_size])
        image = image/(0.5*255.0)-1.0  # normalize to [-1,1] range
        return image

    def load_and_preprocess_image(self, path):
        image = tf.read_file(path)
        return self.preprocess_image(image, isCrop = True)

    def get_iterator(self):
        AUTOTUNE = tf.contrib.data.AUTOTUNE

        all_image_paths = list(self.data_root.glob('*'))#for mnist(has sub-directories) uses */* 
        all_image_paths = [str(path) for path in all_image_paths]
        random.shuffle(all_image_paths)
        image_count = len(all_image_paths)
        print('image_count:', image_count)
        path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
        image_ds = path_ds.map(self.load_and_preprocess_image, num_parallel_calls=4)
        print('image shape: ', image_ds.output_shapes)
        print('types: ', image_ds.output_types)
        print(image_ds)


        #ds = image_label_ds.shuffle(buffer_size=image_count)
        ds = image_ds.repeat()
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        iterator = ds.make_initializable_iterator()
        return iterator, image_count


    
    