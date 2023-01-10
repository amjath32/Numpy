

import numpy as np

# 1.printing version & configuration:

import numpy as np

print(np.__version__)

import numpy as np

np.__config__.show()

# 2.creating a null vector of size 10:

null_vector = np.zeros(10)

print(null_vector)

# 3.finding the memory & size of an array:

arr = np.array([[1, 2, 3], [4, 5, 6]])
memory_size = arr.itemsize * arr.size

print(memory_size)

# 4.documentation of the numpy add function from the command line:

np.info(np.add)

# 5.Creating a null vector of size 10 but the fifth value which is 1:

null_vector = np.zeros(10)
null_vector[4] = 1

print(null_vector)

# 6.Creating a vector with values ranging from 10 to 49:

vector = np.arange(10, 50)
print(vector)

# 7.reversing a vector:

vector = np.array([1, 2, 3, 4, 5])
reversed_vector = vector[::-1]
print(reversed_vector)

# 8.Creating a 3x3 matrix with values ranging from 0 to 8:

matrix = np.arange(9).reshape((3, 3))
print(matrix)

# 9.Finding indices of non-zero elements from [1,2,0,0,4,0]:

array = np.array([1, 2, 0, 0, 4, 0])
indices = np.nonzero(array)
print(indices)

# 10.Creating a 3x3 identity matrix:

identity_matrix = np.eye(3)
print(identity_matrix)

# 11.Creating a 3x3x3 array with random values:

array = np.random.random((3, 3, 3))
print(array)

# 12.Creating a 10x10 array with random values and finding the minimum and maximum values:

array = np.random.random((10, 10))
min_value = array.min()
max_value = array.max()
print(min_value)
print(max_value)

# 13.Creating a random vector of size 30 and finding the mean value:

vector = np.random.random(30)
mean = vector.mean()
print(mean)

# 14.Creating a 2d array with 1 on the border and 0 inside:

array = np.ones((5, 5))
array[1:-1, 1:-1] = 0
print(array)

# 15.adding a border (filled with 0's) around an existing array:

array = np.array([[1, 2, 3], [4, 5, 6]])
padded_array = np.pad(array, pad_width=1, mode='constant', constant_values=0)
print(padded_array)

# 16.the result of the following expression is:

var = 0 * np.nan  # nan
np.nan == np.nan  # False
np.inf > np.nan  # False
np.nan - np.nan  # nan
0.3 == 3 * 0.1  # True

# 17.Creating a 5x5 matrix with values 1,2,3,4 below the diagonal:

matrix = np.diag([1, 2, 3, 4, 5])
matrix[1:, :-1] = 1
print(matrix)

# 18.Creating a 8x8 matrix and filling it with a checkerboard pattern:

matrix = np.zeros((8, 8), dtype=int)
matrix[1::2, ::2] = 1
matrix[::2, 1::2] = 1
print(matrix)

# 19.Considering a (6,7,8) shape array, what is the index (x,y,z) of the 100th element:

array = np.zeros((6, 7, 8))
indices = np.unravel_index(100, array.shape)
print(indices)

# 20.Creating a checkerboard 8x8 matrix using the tile function:

pattern = np.array([[0, 1], [1, 0]])
matrix = np.tile(pattern, (4, 4))
print(matrix)

# 21.Normalizing a 5x5 random matrix:

matrix = np.random.rand(5, 5)
min = matrix.min(axis=0)
max = matrix.max(axis=0)
range = max - min
normalized_matrix = (matrix - min) / range
print(normalized_matrix)

# 22.Creating a custom dtype that describes a color as four unsigned bytes (RGBA):

color_dtype = np.dtype([("r", np.uint8), ("g", np.uint8), ("b", np.uint8), ("a", np.uint8)])
colors = np.array([(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)], dtype=color_dtype)

print(colors)
print(colors["r"])
print(colors["g"])
print(colors["b"])
print(colors["a"])

# 23.Multiplying a 5x3 matrix by a 3x2 matrix (real matrix product):

matrix_a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
matrix_b = np.array([[1, 2], [3, 4], [5, 6]])
matrix_product = np.dot(matrix_a, matrix_b)
print(matrix_product)

# 24.Given a 1D array, negating all elements which are between 3 and 8, in place:

array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
mask = np.logical_and(array > 3, array <= 8)
array[mask] = -array[mask]
print(array)

# 25.output of the following script is:

# print(sum(range(5),-1))
# from numpy import *
# print(sum(range(5),-1))

# 9
# 10

# 26.Considering an integer vector Z, which of these expressions are legal?:

'''Z ** Z
2 << Z >> 2
Z < - Z  # is not legal
1j * Z
Z / 1 / 1
Z < Z > Z  # is not legal'''

# 27.the result of the following expressions are:

'''np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)

#the result is nan'''

# 28.round away from zero a float array:

array = np.random.uniform(-10, 10, size=10)

sign_array = np.copysign(np.ones_like(array), array)

rounded_array = np.ceil(np.abs(array))

abs_rounded_array = np.abs(rounded_array)

result = np.multiply(abs_rounded_array, sign_array)

print(result)

# 29.to find common values between two arrays:

array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([3, 4, 5, 6, 7])

common_values = np.intersect1d(array1, array2)

print(common_values)

# 30.How to ignore all numpy warnings:

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

np.seterr(divide="warn")
np.array([1]) / np.array([0])

# 31.Is the following expressions true?

print="true"

# 32.to get the dates of yesterday, today and tomorrow:

today = np.datetime64('today')
print("today")

yesterday = today - np.timedelta64(1, 'D')
print("yesterday")

tomorrow = today + np.timedelta64(1, 'D')
print("tomorrow")

# 33.How to get all the dates corresponding to the month of July 2016?

start_date = np.datetime64('2016-07-01')

end_date = np.datetime64('2016-07-31')

dates = np.arange(start_date, end_date+1, dtype='datetime64[D]')

print(dates)

# 34. How to compute ((A+B)*(-A/2)) in place (without copy)?


A = np.array([1, 2, 3, 4, 5])
B = np.array([5, 4, 3, 2, 1])

np.add(A, B, out=A)

np.negative(A, out=A)

np.divide(A, 2, out=A)

np.multiply(A, B, out=A)

print(A)

# 35.Extracting the integer part of a random array using 5 different methods:

# %

a = np.random.random(5)
print(a)

# Extract the integer part using the modulo operator
b = a - a % 1
print(b)

#np.floor()

a = np.random.random(5)
print(a)

# Extract the integer part using np.floor()
b = np.floor(a)
print(b)

#np.ceil()

a = np.random.random(5)
print(a)

# Extract the integer part using np.ceil()
b = np.ceil(a) - 1
print(b)

#astype()

a = np.random.random(5)
print(a)

# Extract the integer part using astype()
b = a.astype(int)
print(b)

# 36.Creating a 5x5 matrix with row values ranging from 0 to 4:

a = np.arange(5)

matrix = a.reshape((5, 5))

print(matrix)

# 37.Consider a generator function that generates 10 integers and use it to build an array:

def generator():
    for i in range(10):
        yield i

a = np.fromiter(generator(), dtype=int)

print(a)

# 38.Creating a vector of size 10 with values ranging from 0 to 1, both excluded:

a = np.linspace(0, 1, num=10, endpoint=False)

print(a)

# 39.Creating a random vector of size 10 and sort it:

a = np.random.random(10)
print(a)

a.sort()
print(a)

# 40.How to sum a small array faster than np.sum?

a = np.array([1, 2, 3, 4])

result = np.add.reduce(a)

print(result)

# 41.Consider two random array A and B, check if they are equal:

A = np.random.random((3, 3))
B = np.random.random((3, 3))

result1 = np.allclose(A, B)
print(result1)

result2 = np.array_equal(A, B)
print(result2)

# 42.Make an array immutable (read-only):

a = np.array([1, 2, 3, 4])

a.flags.writeable = False

try:
    a[0] = 10
except ValueError:

    print("Cannot modify read-only array")

# 43.Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates:

cartesian = np.random.random((10, 2))

r = np.sqrt(cartesian[:, 0]**2 + cartesian[:, 1]**2)
theta = np.arctan2(cartesian[:, 1], cartesian[:, 0])

print(r)
print(theta)

# 44.Create random vector of size 10 and replace the maximum value by 0:

vec = np.random.random(10)

idx = np.argmax(vec)

vec[idx] = 0

print(vec)

# 45.create a structured array with x and y coordinates covering the [0,1]x[0,1] area:

x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))

dt = np.dtype([('x', float), ('y', float)])

coords = np.array([(xx, yy) for xx, yy in zip(x.flatten(), y.flatten())], dtype=dt)

print(coords)

# 46.Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj)):

X = np.random.random(5)
Y = np.random.random(5)

diffs = np.subtract.outer(X, Y)

recip = np.reciprocal(diffs)

C = np.triu(recip)

print(C)

# 47.Print the minimum and maximum representable value for each numpy scalar type:

for t in [np.int8, np.int16, np.int32, np.int64]:
    info = np.iinfo(t)
    print(f"{t}: {info.min} to {info.max}")

for t in [np.float16, np.float32, np.float64, np.float128]:
    info = np.finfo(t)
    print(f"{t}: {info.min} to {info.max}, precision {info.precision} digits, resolution {info.resolution}")

# 48.How to print all the values of an array?

a = np.random.random((3,3))

np.set_printoptions(threshold=np.inf)

print(a)

# 49. How to find the closest value (to a given scalar) in a vector?

v = np.random.random(10)

scalar = 0.5

idx = np.abs(v - scalar).argmin()

closest_value = v[idx]

print(f"The closest value to {scalar} in the vector is {closest_value}")

# 50.Create a structured array representing a position (x,y) and a color (r,g,b):

dt = np.dtype([('position', [('x', 'float'), ('y', 'float')]),
               ('color', [('r', 'uint8'), ('g', 'uint8'), ('b', 'uint8')])])

s = np.array([((0.1, 0.2), (255, 0, 0)),
              ((0.3, 0.4), (0, 255, 0)),
              ((0.5, 0.6), (0, 0, 255))], dtype=dt)

print(s)

# 51.Consider a random vector with shape (100,2) representing coordinates, find point by point distances:

a = np.random.random((100,2))
b = np.random.random((100,2))

a = np.atleast_2d(a)
b = np.atleast_2d(b)

b = b.T

distances = np.sqrt(np.sum((a-b)**2, axis=0))

print(distances)

# 52.How to convert a float (32 bits) array into an integer (32 bits) in place?

a = np.random.random(10)

a = a.astype(np.int32, copy=False)

print(a)

# 53.How to read the following file?

data = np.genfromtxt('data.txt', delimiter=',', dtype=None, missing_values=' ', filling_values=0)

print(data)

# 54.What is the equivalent of enumerate for numpy arrays?

a = np.array([[1, 2, 3], [4, 5, 6]])

for i, x in np.ndenumerate(a):
    print(i, x)

# 55.Generate a generic 2D Gaussian-like array:

mean = [0, 0]
std = [1, 1]

x, y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))

z = np.exp(-((x - mean[0]) ** 2 + (y - mean[1]) ** 2) / (2 * std[0] ** 2))

# 56.How to randomly place p elements in a 2D array?

m, n = 10, 10

p = 5

array = np.zeros((m, n))

indices = np.random.choice(m * n, p, replace=False)

np.put(array, indices, 1)

# 57.Subtract the mean of each row of a matrix:

row_means = A.mean(axis=1, keepdims=True)

A_centered = A - row_means

# 58.How to sort an array by the nth column?

# Sort the array by the nth column
sorted_indices = np.argsort(A[:,n])
A_sorted = A[sorted_indices]

# 59.How to tell if a given 2D array has null columns?

has_null_columns = ~np.any(A, axis=0)

null_columns = np.arange(A.shape[1])[has_null_columns]
print(null_columns)

# 60.Find the nearest value from a given value in an array:

idx = np.abs(A - v).argmin()

nearest_value = A.flat[idx]

# 61.Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator?

a = np.array([[1, 2, 3]])
b = np.array([[4], [5], [6]])

it = np.nditer([a, b, None])

for x, y, z in it:
    z[...] = x + y

print(it.operands[2])

# 62.Create an array class that has a name attribute:

import numpy as np

class NamedArray(np.ndarray):
    def __new__(cls, input_array, name=""):
        # Create the array and set the name attribute
        obj = np.asarray(input_array).view(cls)
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        # Set the name attribute for any new instances
        if obj is None:
            return
        self.name = getattr(obj, 'name', "")

a = NamedArray([1, 2, 3], name="my_array")

print(a.name)

b = a[::-1]

print(b.name)

# 63.Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)

vec = np.array([1, 2, 3, 4, 5])

indices = np.array([0, 2, 3, 0, 2])

# Use np.bincount to count the number of occurrences of each index
counts = np.bincount(indices)

# Use np.add.at to add 1 to each element at the specified indices
np.add.at(vec, indices, 1)

vec /= counts

print(vec)  # [2.0, 2.0, 4.0, 5.0, 5.0]

# 64.How to accumulate elements of a vector (X) to an array (F) based on an index list (I)?

X = np.array([1, 2, 3, 4, 5])

I = np.array([0, 2, 3, 0, 2])

F = np.zeros(3)

np.bincount(I, weights=X, minlength=3)

print(F)  # [9.0, 0.0, 7.0]

# 65.Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors:

image = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]], dtype='uint8')

image_flat = image.reshape(-1, 3)

unique, counts = np.unique(image_flat, return_counts=True, axis=0)

print(unique.shape[0])  # 4

# 66.Considering a four dimensions array, how to get sum over the last two axis at once?

arr = np.random.randint(0, 10, size=(2, 3, 4, 5))
result = arr.sum(axis=(-2,-1))
print(result.shape)

# 67.Considering a one-dimensional vector D, how to compute means of subsets of D using a
# vector S of same size describing subset indices?


D = np.array([3, 4, 5, 6, 7, 8])
S = np.array([0, 0, 1, 1, 1, 2])

counts = np.bincount(S)

sums = np.bincount(S, weights=D)

means = sums / counts
print(means)  # Output: [3. 5. 8.]

# 68.How to get the diagonal of a dot product?

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

dot_product = np.dot(A, B)

diagonal = np.diag(dot_product)

print(diagonal)  # Output: [26 34]

# 69.Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value?

vec = np.array([1, 2, 3, 4, 5])

new_vec = np.repeat(vec, 4)
new_vec[3::4] = 0

print(new_vec)

# 70.Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)?

arr1 = np.ones((5, 5, 3))

arr2 = np.ones((5, 5))

arr2_expanded = arr2[:, :, None]

result = arr1 * arr2_expanded

print(result.shape)  # prints (5, 5, 3)

# 71.How to swap two rows of an array?

arr = np.arange(25).reshape(5, 5)

arr[[0, 1]] = arr[[1, 0]]

print(arr)

# 72.Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments
# composing all the triangles:


triangles = np.array([[[0, 0], [1, 0], [0, 1]],
                      [[2, 0], [3, 0], [2, 1]],
                      [[4, 0], [5, 0], [4, 1]],
                      [[6, 0], [7, 0], [6, 1]],
                      [[8, 0], [9, 0], [8, 1]],
                      [[10, 0], [11, 0], [10, 1]],
                      [[12, 0], [13, 0], [12, 1]],
                      [[14, 0], [15, 0], [14, 1]],
                      [[16, 0], [17, 0], [16, 1]],
                      [[18, 0], [19, 0], [18, 1]]])


line_segments = np.repeat(triangles, 3, axis=1)

line_segments = np.roll(line_segments, shift=-1, axis=1)

line_segments = np.sort(line_segments, axis=1)

# 73. Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C?

def create_array_from_bincount(C):
    A = []
    for i, c in enumerate(C):
        A += np.repeat(i, c).tolist()
    return np.array(A)

# 74.How to compute averages using a sliding window over an array?

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

window_size = 3

rolling_avg = (np.cumsum(arr) / window_size)[window_size - 1:] / window_size

print(rolling_avg)

# 75.Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) and
# each subsequent row is shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1]):

import numpy as np
from numpy.lib import stride_tricks

# Input array
Z = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

shape = (3, len(Z) - 2)
strides = (Z.itemsize, Z.itemsize)
A = stride_tricks.as_strided(Z, shape=shape, strides=strides)

print(A)

# 76.to negate a boolean, or to change the sign of a float inplace:

a = np.array([True, False, True])

np.logical_not(a, out=a)

print(a)  # Output: [False, True, False]

#np.negative

a = np.array([1.0, -2.0, 3.0])

np.negative(a, out=a)

print(a)  # Output: [-1.0, 2.0, -3.0]

# 77.Consider 2 sets of points P0,P1 describing lines (2d) and a point p,
# how to compute distance from p to each line i (P0[i],P1[i])?

#To compute the distance for multiple lines defined by points P0 and P1,
#you can use a loop or vectorize the formula using NumPy functions. For example:

P0_x, P0_y = p[:,0], p[:,1]
P1_x, P1_y = p[:,0], p[:,1]
p_x, p_y =p[0], p[1]

distances = np.abs((P1_y - P0_y) * p_x - (P1_x - P0_x) * p_y + P1_x * P0_y - P1_y * P0_x) / \
            np.sqrt((P1_y - P0_y)**2 + (P1_x - P0_x)**2)

# 78.Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P,
# how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i])?

distances[i, J] = abs((p[i, 1] - p[i, 1]) * p[J, 0] - (p[i, 0] - p[i, 0]) * p[J, 1] + p[i, 0] *
                     p[i, 1] - p[i, 1] * p[i, 0]) / np.sqrt((p[i, 1] - p[i, 1])**2 + (p[i, 0] - p[i, 0])**2)

# 79.Consider an arbitrary array, write a function that extract a subpart with a fixed shape and
# centered on a given element (pad with a fill value when necessary):


def extract_subpart(arr, center, shape, fill_value):
    # Calculate start and end indices for each dimension
    start_idx = [max(0, c - s // 2) for c, s in zip(center, shape)]
    end_idx = [min(c + s // 2 + 1, arr.shape[i]) for i, (c, s) in enumerate(zip(center, shape))]

    # Create a view of the original array with the desired shape
    subarr = np.array(arr)[tuple(slice(start, end) for start, end in zip(start_idx, end_idx))]

    # Create a new array with the desired shape, filled with the fill value
    result = np.full(shape, fill_value, dtype=arr.dtype)

    # Calculate the start and end indices for the subarray in the result array
    start_idx = [max(0, s // 2 - c) for c, s in zip(center, shape)]
    end_idx = [min(s // 2 + c + 1, shape[i]) for i, (c, s) in enumerate(zip(center, shape))]

    # Copy the subarray into the result array
    result[tuple(slice(start, end) for start, end in zip(start_idx, end_idx))] = subarr

    return result

# 80.Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
# how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]?

import numpy as np
from numpy.lib import stride_tricks

Z = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])

# Get the shape and strides of the input array
shape = Z.shape
strides = Z.strides

# Set the shape and strides for the output array
n = 4 # number of elements in each subarray
new_shape = (len(Z) - n + 1, n)
new_strides = (strides[0], strides[0])

# Generate the output array using as_strided
R = stride_tricks.as_strided(Z, shape=new_shape, strides=new_strides)

print(R)
# [[ 1  2  3  4]
#  [ 2  3  4  5]
#  [ 3  4  5  6]
#  [ 4  5  6  7]
#  [ 5  6  7  8]
#  [ 6  7  8  9]
#  [ 7  8  9 10]
#  [ 8  9 10 11]
#  [ 9 10 11 12]
#  [10 11 12 13]
#  [11 12 13 14]]

# 81.Compute a matrix rank:

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

_, s, _ = np.linalg.svd(A)

rank = np.sum(s > 1e-10)

print(rank)  # Output: 2

# 82.How to find the most frequent value in an array?

import numpy as np

arr = np.random.randint(0, 10, size=20)

counts = np.bincount(arr)

most_frequent_index = np.argmax(counts)

most_frequent_value = arr[most_frequent_index]

print(f"The most frequent value in the array is {'most_frequent_value'}")

# 83. Extract all the contiguous 3x3 blocks from a random 10x10 matrix:

matrix = np.random.rand(10, 10)

block_size = (3, 3)
strides = matrix.strides
blocks = np.lib.stride_tricks.as_strided(matrix, block_size + matrix.shape, strides * 2)

print(blocks)

# 84.Create a 2D array subclass such that Z[i,j] == Z[j,i]:

import numpy as np

class SymmetricArray(np.ndarray):
    def __setitem__(self, index, value):
        i, j = index
        super().__setitem__((i, j), value)
        super().__setitem__((j, i), value)

Z = SymmetricArray((3, 3))
Z[0, 1] = 1
print(Z[1, 0])  # prints 1

# 85.Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1).
# How to compute the sum of of the p matrix products at once? (result has shape (n,1)):

result = np.tensordot(m, v, axes=([1], [0]))

# 86.Considering a 16x16 array, how to get the block-sum (block size is 4x4):

import numpy as np

arr = np.random.rand(16, 16)

block_sum = np.add.reduceat(np.add.reduceat(arr, np.arange(0, arr.shape[0], 4), axis=0),
                            np.arange(0, arr.shape[1], 4), axis=1)

print(block_sum)

# 87.How to implement the Game of Life using numpy arrays?

import numpy as np

cells = np.random.randint(2, size=(16, 16))

def update_state(cells):
  # Get the 3x3 subarrays centered on each cell
  neighbors = cells[1:-1, 1:-1]
  top_neighbors = cells[:-2, 1:-1]
  bottom_neighbors = cells[2:, 1:-1]
  left_neighbors = cells[1:-1, :-2]
  right_neighbors = cells[1:-1, 2:]

  num_live_neighbors = np.sum(neighbors) + np.sum(top_neighbors) + np.sum(bottom_neighbors) + \
                       np.sum(left_neighbors) + np.sum(right_neighbors)

  # Apply the rules of the Game of Life to each cell
  cells[1:-1, 1:-1] = np.where((cells[1:-1, 1:-1] == 1) & ((num_live_neighbors == 2) | (num_live_neighbors == 3)), 1, 0)
  cells[1:-1, 1:-1] = np.where(cells[1:-1, 1:-1] == 0) & (num_live_neighbors == 3), 1, cells[1:-1, 1]

# 88.to get the n largest values of an array:

import numpy as np

  # Sample array
arr = np.array([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])

  # Get the indices that would sort the array in ascending order
indices = np.argsort(arr)

  # Use the indices to index the original array and get the last n elements, which are the n largest values
largest_values = arr[indices[-n:]]

# 89.Given an arbitrary number of vectors, build the cartesian product (every combinations of every item):

  # Define the vectors
v1 = [1, 2, 3]
v2 = ['a', 'b', 'c']
v3 = ['x', 'y', 'z']

  # Build the cartesian product
indices = np.indices((len(v1), len(v2), len(v3)))
cartesian_product = np.stack([v1[indices[0]], v2[indices[1]], v3[indices[2]]], axis=-1)

  # Print the result
print(cartesian_product)

# 90.How to create a record array from a regular array?

import numpy as np

  # Create a regular array with 3 rows and 2 columns
data = np.array([[1, 2], [3, 4], [5, 6]])

  # Create the record array
record_array = np.core.records.fromarrays(data.T, names='a,b', formats='i4,i4')

print(record_array)

# 91.Consider a large vector Z, compute Z to the power of 3 using 3 different methods:

np.power();

result = np.power(Z, 3)

result = Z * Z * Z

np.einsum:\
   result = np.einsum('i,i,i->i', Z, Z, Z)

# 92.Consider two arrays A and B of shape (8,3) and (2,2).
  # How to find rows of A that contain elements of each row of B regardless of the order of the elements in B?

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]])
B = np.array([[1, 2], [3, 4]])

indices = np.where(np.all(np.isin(A, B), axis=1))[0]

  # Print the rows of A that satisfy the condition
print(A[indices])

# 93.Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3]):

import numpy as np

matrix = np.array([[2, 2, 3],
                     [1, 2, 2],
                     [2, 2, 2],
                     [3, 3, 3],
                     [1, 1, 1]])

unequal_rows = matrix[np.any(matrix != matrix[:, 0][:, None], axis=1)]

print(unequal_rows)

# 94.Convert a vector of ints into a matrix binary representation:

binary = np.unpackbits(x, axis=1)

# 95.Given a two dimensional array, how to extract unique rows?

import numpy as np

  # Create a 2D array with duplicate rows
arr = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3], [7, 8, 9]])

  # Extract unique rows
unique_rows = np.unique(arr, axis=0)

print(unique_rows)  # Output: [[1 2 3] [4 5 6] [7 8 9]]

# 96.Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function:

#For inner product:
np.einsum('i,i->', A, B)
#For outer product:
np.einsum('i,j->ij', A, B)
#for element wise sum:
np.einsum('i,i->i', A, B)
#for element wise multiplication:
np.einsum('i,i->i', A, B)

# 97.Considering a path described by two vectors (X,Y), how to sample it using equidistant samples:


# 98.Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial
  # distribution with n degrees, i.e., the rows which only contain integers and which sum to n:

import numpy as np

X_int = X.astype(int)

is_int = np.equal(X, X_int).all(axis=1)

row_sum = np.sum(X_int, axis=1)
is_sum_n = np.equal(row_sum, n)

X[np.logical_and(is_int, is_sum_n)]

# 99. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X
#(i.e., resample the elements of an array with replacement N times, compute the mean of each sample,
#and then compute percentiles over the means):

N = 1000

bootstrapped_means = [np.mean(np.random.choice(X, size=len(X), replace=True)) for _ in range(N)]
  
lower, upper = np.percentile(bootstrapped_means, [2.5, 97.5])

print(f'95% confidence interval: [{lower}, {upper}]')





























































































