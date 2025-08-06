# NumPy - Professional README

## 📖 Overview

NumPy (Numerical Python) is an essential Python library used for scientific and numerical computations. It allows efficient operations on large multi-dimensional arrays and matrices and provides a comprehensive collection of mathematical functions.

---

## 📦 Installation

```bash
pip install numpy
```

---

## 🔧 Creating Arrays

### ✅ 1D, 2D, and 3D Arrays

```python
import numpy as np

arr_1d = np.array([1, 2, 3])
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
```

### 📃 From Python Lists

```python
arr = np.array([10, 20, 30])
```

### ⚙️ Special Arrays

```python
np.zeros((2, 3))       # Array of zeros
np.ones((3, 2), int)    # Array of ones with int type
np.full((2, 2), 9)      # Filled with custom number
```

### 🔢 Sequences

```python
np.arange(1, 10, 2)     # Range with step
np.linspace(1, 10, 5)   # Evenly spaced numbers
np.eye(3)               # Identity matrix
```

---

## 📊 Array Properties

```python
arr.shape   # Shape of array
arr.size    # Number of elements
arr.ndim    # Number of dimensions
arr.dtype   # Data type
```

### 🔄 Type Conversion

```python
arr.astype(float)
```

---

## ➕ Arithmetic & Aggregations

```python
arr + 2
arr * 5
arr ** 2

np.sum(arr)
np.mean(arr)
np.min(arr)
np.max(arr)
np.std(arr)
np.var(arr)
```

---

## 🎯 Indexing, Slicing, Masking

### 🔍 Basic Indexing

```python
arr[0], arr[-1], arr[1:4], arr[::2], arr[::-1]
```

### 🧠 Fancy Indexing

```python
arr[[0, 2, 4]]
```

### ✅ Boolean Masking

```python
arr[arr > 10]
```

---

## 🔁 Reshaping and Flattening

```python
arr.reshape(3, 2)
arr.ravel()       # Returns flattened array (view)
arr.flatten()     # Returns flattened array (copy)
```

---

## ✂️ Modify Arrays

### ➕ Insert

```python
np.insert(arr, 2, 100)
np.insert(arr_2d, 1, [10, 20, 30], axis=0)
```

### ➕ Append

```python
np.append(arr, [70, 80])
```

### 🔗 Concatenate

```python
np.concatenate((arr1, arr2))
```

### ❌ Delete

```python
np.delete(arr, 1)
np.delete(arr_2d, 0, axis=0)
```

---

## 🧱 Stacking and Splitting

```python
np.vstack((arr1, arr2))   # Vertical stack
np.hstack((arr1, arr2))   # Horizontal stack

np.split(arr, 3)          # Equal split
np.vsplit(arr_2d, 2)
np.hsplit(arr_2d, 2)
```

---

## ⚡ Broadcasting & Vectorization

### 📈 Broadcasting Example

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
vector = np.array([10, 20, 30])
matrix + vector
```

### 🚀 Vectorization vs Loop

```python
# Traditional loop
[x + y for x, y in zip(list1, list2)]

# NumPy vectorized
arr1 + arr2
```

---

## 🚫 Missing, NaN & Infinite Values

### ❓ NaN Handling

```python
np.isnan(arr)
np.nan_to_num(arr, nan=0)
```

### ♾️ Infinite Handling

```python
np.isinf(arr)
np.nan_to_num(arr, posinf=999, neginf=-999)
```

---

## 💡 Tips

* Use `reshape` to change the shape of arrays without changing data.
* Use `broadcasting` to apply operations between arrays of different shapes.
* Prefer `vectorized` operations over loops for performance.
* Always check `shape` before doing arithmetic operations on multiple arrays.

---

## 📚 Use in Data Science

NumPy is at the core of scientific computing in Python and is heavily used in:

* Data cleaning (missing values, reshaping)
* Machine Learning (input transformation)
* Libraries like Pandas, Scikit-learn, TensorFlow, and PyTorch

---

**Author:** *ifra jabeen*
