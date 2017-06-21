# REF [file] >> ${KERAS_HOME}/keras/preprocessing/image.py.

"""Fairly basic set of tools for real-time data augmentation on image data.
Can easily be extended to include new transformations,
new preprocessing methods, etc...
"""
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from six.moves import range
import os

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator, DirectoryIterator, array_to_img, img_to_array, load_img


class ImageDataGeneratorWithCrop(ImageDataGenerator):
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 random_crop_size=None,
                 center_crop_size=None,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None):
        super().__init__(featurewise_center,
                 samplewise_center,
                 featurewise_std_normalization,
                 samplewise_std_normalization,
                 zca_whitening,
                 zca_epsilon,
                 rotation_range,
                 width_shift_range,
                 height_shift_range,
                 shear_range,
                 zoom_range,
                 channel_shift_range,
                 fill_mode,
                 cval,
                 horizontal_flip,
                 vertical_flip,
                 rescale,
                 preprocessing_function,
                 data_format)
        self.random_crop_size = random_crop_size
        self.center_crop_size = center_crop_size

    # REF [site] >> https://github.com/fchollet/keras/issues/3338
    def random_crop(self, x):
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        h, w = x.shape[img_row_axis], x.shape[img_col_axis]
        if 'channels_first' == self.data_format:
            rangeh = (h - self.random_crop_size[img_row_axis - 1]) // 2
            rangew = (w - self.random_crop_size[img_col_axis - 1]) // 2
        else:
            rangeh = (h - self.random_crop_size[img_row_axis]) // 2
            rangew = (w - self.random_crop_size[img_col_axis]) // 2
        offsetw = 0 if rangew == 0 else np.random.randint(rangew)
        offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
        if 'channels_first' == self.data_format:
            return x[:, offseth:offseth+self.random_crop_size[img_row_axis - 1], offsetw:offsetw+self.random_crop_size[img_col_axis - 1]]
        else:
            return x[offseth:offseth+self.random_crop_size[img_row_axis], offsetw:offsetw+self.random_crop_size[img_col_axis], :]

    # REF [site] >> https://github.com/fchollet/keras/issues/3338
    def center_crop(self, x):
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        centerh, centerw = x.shape[img_row_axis] // 2, x.shape[self.col_axis] // 2
        if 'channels_first' == self.data_format:
            halfh, halfw = self.center_crop_size[img_row_axis - 1] // 2, self.center_crop_size[img_col_axis - 1] // 2
        else:
            halfh, halfw = self.center_crop_size[img_row_axis] // 2, self.center_crop_size[img_col_axis] // 2
        if 'channels_first' == self.data_format:
            return x[:, centerh-halfh:centerh+halfh, centerw-halfw:centerw+halfw]
        else:
            return x[centerh-halfh:centerh+halfh, centerw-halfw:centerw+halfw, :]

    def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='png'):
        return NumpyArrayIteratorWithCrop(
            x, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)

    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False):
        return DirectoryIteratorWithCrop(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links)


class NumpyArrayIteratorWithCrop(NumpyArrayIterator):
    """Iterator yielding data from a Numpy array.

    # Arguments
        x: Numpy array of input data.
        y: Numpy array of targets data.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, x, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png'):
        super().__init__(x, y, image_data_generator,
                 batch_size, shuffle, seed,
                 data_format,
                 save_to_dir, save_prefix, save_format)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.x.shape)[1:]), dtype=K.floatx())
        for i, j in enumerate(index_array):
            x = self.x[j]
            if self.image_data_generator.random_crop_size is not None:
                x = self.image_data_generator.random_crop(x)
            x = self.image_data_generator.random_transform(x.astype(K.floatx()))
            x = self.image_data_generator.standardize(x)
            if self.image_data_generator.center_crop_size is not None:
                x = self.image_data_generator.center_crop(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y


class DirectoryIteratorWithCrop(DirectoryIterator):
    """Iterator capable of reading images from a directory on disk.

    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        classes: Optional list of strings, names of sudirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False):
        super().__init__(directory, image_data_generator,
                 target_size, color_mode,
                 classes, class_mode,
                 batch_size, shuffle, seed,
                 data_format,
                 save_to_dir, save_prefix, save_format,
                 follow_links)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        if self.image_data_generator.center_crop_size is not None:
            if self.color_mode == 'rgb':
                if self.data_format == 'channels_last':
                    batch_x = np.zeros((current_batch_size,) + self.image_data_generator.center_crop_size + (3,), dtype=K.floatx())
                else:
                    batch_x = np.zeros((current_batch_size,3,) + self.image_data_generator.center_crop_size, dtype=K.floatx())
            else:
                if self.data_format == 'channels_last':
                    batch_x = np.zeros((current_batch_size,) + self.image_data_generator.center_crop_size + (1,), dtype=K.floatx())
                else:
                    batch_x = np.zeros((current_batch_size,1,) + self.image_data_generator.center_crop_size, dtype=K.floatx())
        elif self.image_data_generator.random_crop_size is not None:
            if self.color_mode == 'rgb':
                if self.data_format == 'channels_last':
                    batch_x = np.zeros((current_batch_size,) + self.image_data_generator.random_crop_size + (3,), dtype=K.floatx())
                else:
                    batch_x = np.zeros((current_batch_size,3,) + self.image_data_generator.random_crop_size, dtype=K.floatx())
            else:
                if self.data_format == 'channels_last':
                    batch_x = np.zeros((current_batch_size,) + self.image_data_generator.random_crop_size + (1,), dtype=K.floatx())
                else:
                    batch_x = np.zeros((current_batch_size,1,) + self.image_data_generator.random_crop_size, dtype=K.floatx())
        else:
            batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(os.path.join(self.directory, fname),
                           grayscale=grayscale,
                           target_size=self.target_size)
            x = img_to_array(img, data_format=self.data_format)
            if self.image_data_generator.random_crop_size is not None:
                x = self.image_data_generator.random_crop(x)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            if self.image_data_generator.center_crop_size is not None:
                x = self.image_data_generator.center_crop(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y
