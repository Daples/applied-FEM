from utils.test_space import Tests

L = 2

p = 1
k = 0
m = 4
neval = 20

# Test B
filename = "test_B.pdf"
spacing_func = lambda i: i * L / m
Tests.test_space(p, k, m, spacing_func, neval, filename)

# Test C
filename = "test_C.pdf"
spacing_func = lambda i: i**2 * L / m**2
Tests.test_space(p, k, m, spacing_func, neval, filename)

# Test D
filename = "test_D.pdf"
spacing_func = lambda i: i * L / m
p = 2
Tests.test_space(p, k, m, spacing_func, neval, filename)

# Test E
filename = "test_E.pdf"
spacing_func = lambda i: i**2 * L / m**2
Tests.test_space(p, k, m, spacing_func, neval, filename)
