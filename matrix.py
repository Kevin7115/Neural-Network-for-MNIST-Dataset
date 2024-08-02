import random
import time
import numpy as np

def dot_product(x, y):
    if len(x) == len(y):
        return sum([a*b for a, b in zip(x, y)])
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

    def __add__(self, other):
        self.valid(other)
    
        new_vals = []
        for i, row in enumerate(self.vals):
            new_vals.append([col + other.vals[i][j] for j, col in enumerate(row)])

        return Matrix(self.rows, self.cols, new_vals) 

    def __sub__(self, other):
        self.valid(other)
    
        new_vals = []
        for i, row in enumerate(self.vals):
            new_vals.append([col - other.vals[i][j] for j, col in enumerate(row)])

        return Matrix(self.rows, self.cols, new_vals) 

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return self.scale(other)
        
        if not isinstance(other, Matrix):
            return self.mult_vector(other)
        
        self.valid(other, multiplication = True)

        # new_vals = []
        # for i in range(self.rows):
        #     x = []
        #     for j in range(other.cols):
        #         x.append(sum(self.vals[i][k] * other.vals[k][j] for k in range(other.rows)))
        #     new_vals.append(x)
        
        # return Matrix(self.rows, other.cols, new_vals)

        # y = other.transpose()
        # new_vals = []
        # for row in y.vals:
        #     new_vals.append(self.mult_vector(row))
        
        # return Matrix(other.cols, self.rows, new_vals).transpose()

        # Numpy version
        C = np.matmul(np.array(self.vals), np.array(other.vals))
        return Matrix(self.rows, other.cols, C.tolist())


    def mult_vector(self, other):
        return [
            dot_product(row, other) for row in self.vals
        ]
    
    def scale(self, c):
        for i in range(self.rows):
            for j in range(self.cols):
                self.vals[i][j] *= c

    def transpose(self):
        A_t = [
            [self.vals[r][c] for r in range(self.rows)] 
            for c in range(self.cols)
        ]

        return Matrix(self.cols, self.rows, A_t)
    
    def hadamard(self, other):
        new_vals = []
        for i, row in enumerate(self.vals):
            x = [
                val * other.vals[i][j] for j, val in enumerate(row)
            ]
            new_vals.append(x)

        return Matrix(self.rows, self.cols, new_vals)

    def __str__(self):
        x = ""
        for i, row in enumerate(self.vals):
            if i == len(self.vals)-1:
                x += f"{row}"
            else:
                x += f"{row}\n"

        return x



# e = Matrix(3, 3, [[1, 2, 1], [0, 1, 0], [2, 3, 4]])
# a = Matrix(3, 2, [[2, 5], [6, 7], [1, 8]])
# res = e*a
# test = [[15, 27], [6, 7], [26, 63]]
# print(res, res.vals == test, "\n") # [[15, 27], [6, 7], [26, 63]]

# e = Matrix(3, 4, [[1, 2, 1, 1], [0, 1, 0, 10], [2, 3, 4, 40]])
# a = Matrix(4, 3, [[2, 5, 7], [6, 7, 1], [1, 8, 3], [1, 2, 1]])
# res = e*a
# test = [[16, 29, 13], [16, 27, 11], [66, 143, 69]]
# print(res, res.vals == test, "\n") # [[16, 29, 13], [16, 27, 11], [66, 143, 69]]

# e = Matrix(4, 6, [[1, 2, 1, 1, -1, 1], [0, 1, 0, 10, -4, 1], [2, 3, 4, 40, -8, 1], [13, 12, 11, 10, -10, 1]])
# a = Matrix(6, 3, [[2, 5, 7], [6, 7, 1], [1, 8, 3], [1, 2, 0], [0, 1, 1], [-5, -10, -15]])
# res = e*a
# test = [[11, 18, -4], [11, 13, -18], [61, 125, 6], [114, 237, 111]]
# print(res, res.vals == test, "\n") # [[11, 18, -4], [11, 13, -18], [61, 125, 6], [114, 237, 111]]

# start = time.time()
# x = Matrix(2000, 2000, randomizer=True)
# z = Matrix(2000, 2000, randomizer = True)
# y = x*z
# end = time.time()
# print(y.cols, end-start)

# A = np.random.random((2, 2))
# B = np.random.random((2, 2))
# C = np.matmul(A, B)
# x = [[1, 2], [3, 4]]
# y = np.matmul(A, x)
# print(y)

# x = [[1], [1]]

# X = Matrix(len(x), 1, x)

# y = [[3], [3]]

# Y = Matrix(len(y), 1, y)
# Z = (X+Y)
# Z*(1/4)
# print(Z)


