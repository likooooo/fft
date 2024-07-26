#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import numpy as np

image_path = sys.argv[1]
image_width = int( sys.argv[2] )
image_height = int( sys.argv[3] )
image_data = np.zeros([image_width, image_height], dtype= 'f8' )

def load_image_data_in_text(data_file, width, height):
    if not data_file.endswith(".txt"):
        raise ValueError("invalid text file name: {}".format(data_file))
    data = np.loadtxt(data_file, dtype=float)
    expected_length = height
    if len(data) != expected_length:
        raise ValueError("data length: {}, but {}*{} is {}".format(len(data), width, height, expected_length))
    return data
def load_Image_data_in_bin(data_file, width, height):
    if not data_file.endswith(".exact"):
        raise ValueError("invalid binary file name: {}".format(data_file))
    data = np.fromfile(data_file, dtype=np.float32)
    expected_length = width*height
    if len(data) != expected_length:
        raise ValueError("data length: {}, but {}*{} is {}".format(len(data), width, height, expected_length))
    data = data.reshape( height, width )
    return data
# image_data = load_image_data_in_text( image_path, image_width, image_height )
image_data = load_Image_data_in_bin( image_path, image_width, image_height )

def get_title():
    return "%s\n * min : %lf\n * max : %lf\n * avg : %lf " %( image_path, image_data.min(), image_data.max(), image_data.mean() )

plt.matshow( image_data )
plt.title( get_title() )
plt.colorbar()
ax = plt.gca()
ax.xaxis.set_ticks_position('bottom')
ax.invert_yaxis()
plt.show()
