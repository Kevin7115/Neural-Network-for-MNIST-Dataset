import random
import time
import numpy as np
import random

def dot_product(x, y):
    if len(x) == len(y):
        total = 0
        for a, b in zip(x, y):
            total += a*b
        return total
    else:
        raise Exception("Invalid Dot Product")


class Matrix:
    def __init__(self, rows, cols, data = None, randomizer = False):
        self.rows = rows
        self.cols = cols
        self.vals = []

        if not data is None:
            self.adjust_vals(data)
        elif randomizer:
            self.randomizer()
        else:
            for _ in range(self.rows):
                self.vals.append([0 for _ in range(self.cols)])

    def adjust_vals(self, data):
        if isinstance(data, Matrix):
            self.valid(data)
            self.vals = data.vals

        elif len(data) == self.rows:
            for row in data:
                if not len(row) == self.cols:
                    raise Exception("Invalid Adjustment")
            self.vals = data
        else:
            raise Exception("Invalid Adjustment")
     
    def randomizer(self, distr = (-1, 1)):
        self.vals = [
            [random.uniform(*distr) for _ in range(self.cols)]
            for _ in range(self.rows)
        ]

    def valid(self, other, multiplication = False):
        if not isinstance(other, Matrix):
            raise Exception("Other matrix is invalid")

        if multiplication:
            if not self.cols == other.rows:
                raise Exception("Invalid Multiplication Operation")
            return
        
        if not (self.rows == other.rows and self.cols == other.cols):
            raise Exception("Invalid Add/Sub Operation")
    
    def operations(self, other, subtract = False, hadamard = False):
        self.valid(other)

        if subtract:
            op = lambda col, j: col - other.vals[i][j]
        elif hadamard:
            op = lambda col, j: col * other.vals[i][j]
        else:
            op = lambda col, j: col + other.vals[i][j]
    
        new_vals = []
        for i, row in enumerate(self.vals):
            new_vals.append([op(col, j) for j, col in enumerate(row)])

        return Matrix(self.rows, self.cols, new_vals) 
        
    def __add__(self, other):
        return self.operations(other)

    def __sub__(self, other):
        return self.operations(other, subtract=True)

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return self.scale(other)
        
        if not isinstance(other, Matrix):
            return self.mult_vector(other)
        
        self.valid(other, multiplication = True)

        y = other.transpose()
        new_vals = []
        for row in y.vals:
            new_vals.append(self.mult_vector(row))
        
        return Matrix(other.cols, self.rows, new_vals).transpose()

        # # Numpy version
        # C = np.matmul(np.array(self.vals), np.array(other.vals))
        # return Matrix(self.rows, other.cols, C.tolist())

    def mult_vector(self, other):
        return [
            dot_product(row, other) for row in self.vals
        ]
    
    def scale(self, c):
        # mutates matrix! #
        for i in range(self.rows):
            for j in range(self.cols):
                self.vals[i][j] *= c

    def transpose(self):
        A_t = [list(row) for row in zip(*self.vals)]
        return Matrix(self.cols, self.rows, A_t)
    
    def hadamard(self, other):
        return self.operations(other, hadamard=True)

    def converter(self):
        x = []
        for row in self.vals:
            for col in row:
                x.append(col)
        return x
    
    def __str__(self):
        x = ""
        for i, row in enumerate(self.vals):
            if i == len(self.vals)-1:
                x += f"{row}"
            else:
                x += f"{row}\n"

        return x


if __name__ == "__main__":
    pass

