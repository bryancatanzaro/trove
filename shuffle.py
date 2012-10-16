import numpy as np
import fractions

class Array(object):
    def __init__(self, *shape):
        self.d = np.ndarray(shape=shape, dtype=np.int32)
    def __str__(self):
        m, n = self.d.shape
        result = ""
        for row in range(m):
            for col in range(n):
                el = self.d[row, col]
                if el < 10:
                    result += ' '
                if el < 100:
                    result += ' '
                result += str(el) + ' '
            result += '\n'
        return result
    def __getitem__(self, key):
        return self.d[key]
    def __setitem__(self, key, val):
        self.d[key] = val
    def shape(self):
        return self.d.shape

def offset_constant(m, n):
    for i in range(m):
        val = n * i
        if val % m == 1:
            return val / m
        
def rotate_constant(m, n):
    for i in range(m):
        val = n * i
        if val % m == 1:
            return val / n

def offset_p2_constant(m, n):
    for i in range(n):
        val = m * i
        dest_col = val % n
        if dest_col == 2:
            return i
        
def offset_p2_constant_2(m, n):
    for i in range(n/2):
        val = 2 + (m * i)
        if (val % n) == 0:
            return i

def permute_p2_constant(m, n):
    return n % m

def mod_mult_inverse(a, b):
    if fractions.gcd(a, b) != 1:
        raise ValueError("Modular Multiplicative Inverse does not exist")
    for m in range(b):
        if (a * m) % b == 1:
            return m

        
def make_row_array(m, n):
    result = Array(m, n)
    for row in range(m):
        for col in range(n):
            result[row, col] = col + row * n
    return result

def make_col_array(m, n):
    result = Array(m, n)
    for row in range(m):
        for col in range(n):
            result[row, col] = row + col * m
    return result

def col_permute(a, p):
    m, n = a.shape()
    result = Array(m, n)
    for col in range(n):
        for row in range(m):
            result[row, col] = a[p[row], col]
    return result

def col_rotate(a, r):
    m, n = a.shape()
    result = Array(m, n)
    for col, rotation in zip(range(n), r):
        for row in range(m):
            result[row, col] = a[(row + rotation) % m, col]
    return result

def col_subrotate(a, r):
    m, n = a.shape()
    result = Array(m, n)
    for col, rotation in zip(range(n), r):
        for row in range(0, m/2):
            result[row, col] = a[row, col]
            result[row + m/2, col] = a[(row + rotation) % (m/2) + m/2, col]
    return result

def row_shuffle(a, s):
    m, n = a.shape()
    result = Array(m, n)
    for row in range(m):
        for col in range(n):
            result[row, col] = a[row, s[row, col]]
    return result

def col_pair_swap(a, s):
    m, n = a.shape()
    result = Array(m, n)
    for col, swap in zip(range(n), s):
        for row in range(0, m, 2):
            if swap:
                result[row, col] = a[row+1, col]
                result[row+1, col] = a[row, col]
            else:
                result[row, col] = a[row, col]
                result[row+1, col] = a[row+1, col]
    return result

def factor_two_swaps(a):
    m, n = a.shape()
    return map(lambda xi: xi >= n/2, range(n))

def golden_shuffles(a):
    m, n = a.shape()
    result = Array(m, n)
    for row in range(m):
        for col in range(n):
            dest_col = a[row, col] % n
            result[row, dest_col] = col
    return result

def factor_two_shuffles(a):
    m, n = a.shape()
    result = Array(m, n)
    for col in range(n):
        odd = col % 2
        state = ((col / 2) * (offset_p2_constant(m, n))) % (n / 2) 
        row_offset = offset_p2_constant_2(m, n)
        for row in range(0, m, 2):
            if not odd:
                result[row, col] = state
                result[row + 1, col] = state + n / 2
            else:
                result[row + 1, col] = state
                result[row, col] = state + n / 2
            state = (state + (row_offset)) % (n / 2)
    return result

def factor_two_rotates(a):
    m, n = a.shape()
    return map(lambda xi: xi % m, range(n))

def factor_two_permutes(a):
    m, n = a.shape()
    #return range(0, m, 2) + range(1, m, 2) 
    return map(lambda xi: (xi * 4) % m, range(m/2)) + map(lambda xi: ((xi * 4) % m) + 1, range(m/2)) 
    
def factor_two_subrotates(a):
    m, n = a.shape()
    return map(lambda xi: (m/2-1) * (xi % 2), range(n))

def transpose(a):
    swapped = col_pair_swap(a, factor_two_swaps(a))
    shuffled = row_shuffle(swapped, factor_two_shuffles(a))
    rotated = col_rotate(shuffled, factor_two_rotates(a))
    permuted = col_permute(rotated, factor_two_permutes(a))
    subrotated = col_subrotate(permuted, factor_two_subrotates(a))
    return permuted


def r2c_conflict_free(a):
    m, n = a.shape()
    for row in range(m):
        dest_cols = set()
        for col in range(n):
            dest_col = a[row, col] / m
            if dest_col in dest_cols:
                return False
            dest_cols.add(dest_col)
    return True
            
def r2c_odd_rotates(a):
    m, n = a.shape()
    constant = m - rotate_constant(m, n)
    return map(lambda xi: (constant * xi) % m, range(n))
    # for rotation in range(m):
    #     candidate_rotation = map(lambda xi: (rotation* xi) % m, range(n))
    #     if r2c_conflict_free(col_rotate(a, candidate_rotation)):
    #         print("Rotation constant: %s" % rotation)
    #         return candidate_rotation
    #assert(False)

def composite_c2r_prerotate(a):
    m, n = a.shape()
    c = fractions.gcd(m, n)
    return map(lambda xi: xi/(n/c), range(n))

def c2r_golden_shuffles(a):
    m, n = a.shape()
    result = Array(m, n)
    #Sentinels in result
    for row in range(m):
        for col in range(n):
            result[row, col] = -1
    for row in range(m):
        for col in range(n):
            dest_col = a[row, col] % n
            if result[row, dest_col] > -1:
                raise ValueError("Array is unshufflable (%s, %s)" % (row, col))
            result[row, dest_col] = col
    return result

def r2c_golden_shuffles(a):
    m, n = a.shape()
    result = Array(m, n)
    for row in range(m):
        for col in range(n):
            dest_col = a[row, col] / m
            result[row, dest_col] = col
    return result

def r2c_odd_shuffles(a):
    m, n = a.shape()
    result = Array(m, n)
    for col in range(n):
        for row in range(m):
            result[row, col] = (m * col + (n % m * row) % m) % n 
    return result

def r2c_odd_permute(a):
    m, n = a.shape()
    constant = rotate_constant(m, n)
    result = map(lambda xi: (xi * constant) % m, range(m))
    return result

def r2c_transpose(a):
    rotated = col_rotate(a, r2c_odd_rotates(a))
    shuffled = row_shuffle(rotated, r2c_odd_shuffles(a))
    permuted = col_permute(shuffled, r2c_odd_permute(a))
    return permuted

def dest_c2r_col(a):
    m, n = a.shape()
    result = Array(m, n)
    for row in range(m):
        for col in range(n):
            result[row, col] = a[row, col] % n
    return result

def composite_c2r_shuffles(a):
    m, n = a.shape()
    c = fractions.gcd(m, n)
    k = mod_mult_inverse(m/c, n/c)
    result = Array(m, n)
    for col in range(n):
        for row in range(m):
            idx = col + row * (n-1)
            if row > m - c + (col % c):
                idx += m
            result[row, col] = \
                ((idx/c)*k % (n/c) + (idx % c) * (n/c)) % n
    return result

def simple_composite_c2r_shuffles(a):
    m, n = a.shape()
    c = fractions.gcd(m, n)
    k = mod_mult_inverse(m/c, n/c)
    result = Array(m, n)
    for col in range(n):
        idx = col
        for row in range(m):
            result[row, col] = \
                ((idx/c)*k % (n/c) + (idx % c) * (n/c)) % n
            idx += n-1
            if row == m - c + (col % c):
                idx += m

    return result

def composite_c2r_permutes(a):
    m, n = a.shape()
    offset = n % m
    c = fractions.gcd(m, n)
    period = m / c
    # return map(lambda xi: (xi * offset - (xi / period)) % m, range(m))
    
    result = [0] * m
    idx = 0
    for col in range(m):
        result[col] = idx
        idx += offset
        if (col % period) == period - 1:
            idx -= 1
        idx = idx % m
    return result

def composite_c2r(a):
    m, n = a.shape()
    pre_rotated = col_rotate(a, composite_c2r_prerotate(a))
    shuffled = row_shuffle(pre_rotated, composite_c2r_shuffles(a))
    post_rotated = col_rotate(shuffled, map(lambda xi: xi % m, range(n)))
    permuted = col_permute(post_rotated, composite_c2r_permutes(a))
    return permuted

# m = 12
# n = 32
# a = make_col_array(m, n)
# print(composite_c2r(a))

def reverse_permute(p):
    permutes = {}
    for (idx, pdx) in enumerate(p):
        permutes[pdx] = idx
    return [permutes[xi] for xi in range(len(p))]


def composite_r2c_shuffles(a):
    m, n = a.shape()
    c = fractions.gcd(m, n)
    p = n / c
    result = Array(m, n)
    for col in range(n):
        lb = (m * col) % n
        ub = lb + m
        rotate = col / p
        idx = lb + rotate
        for row in range(m):
            result[row, col] = (idx) % n
            idx += 1
            if (idx == ub):
                idx = lb
            

    return result
    

def composite_r2c(a):
    m, n = a.shape()
    c = fractions.gcd(m, n)
    pre_permuted = col_permute(a, reverse_permute(composite_c2r_permutes(a)))
    pre_rotated = col_rotate(pre_permuted, map(lambda xi: m - (xi % m), range(n)))
    shuffled = row_shuffle(pre_rotated, composite_r2c_shuffles(a))
    post_rotated = col_rotate(shuffled, map(lambda xi: m - (xi/(n/c)), range(n)))
    return post_rotated


def simple_r2c_po2_shuffle(a):
    m, n = a.shape()
    result = Array(m, n)
    for col in range(n):
        offset = (m * col + col / (n/m)) % n
        lb = (offset/m)*m 
        for row in range(m):
            result[row, col] = offset
            if (offset == lb):
                offset += m - 1
            else:
                offset -= 1
    return result

m = 16
n = 32
a = make_row_array(m, n)
b = col_rotate(a, [xi % m for xi in range(0, n)])
c = r2c_golden_shuffles(b)
print(b)
print(c)
print simple_r2c_po2_shuffle(a)
# m = 10
# n = 32
# a = make_row_array(m, n)
# print(a)
# print(composite_r2c(a))


        
# a = make_row_array(5, 32)
# print(a)
# print(r2c_odd_shuffles(a))
# b = r2c_transpose(a)
# print(b)

# def check_shuffles(a):
#     shuffles = factor_two_shuffles(a)
#     goldens = golden_shuffles(col_pair_swap(a, factor_two_swaps(a)))
#     print shuffles
#     print goldens

# check_shuffles(a)
