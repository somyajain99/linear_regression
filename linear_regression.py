import numpy as np

def cost_func(computed_y, actual_y):
    assert len(computed_y) == len(actual_y), "length of computed_y and actual_y are not same"
    sum_cost = 0
    for i in range(len(computed_y)):
        sum_cost += (computed_y[i] - actual_y[i])**2
    output_cost = (0.5 * sum_cost)/(len(computed_y))
    return output_cost

def gradient_descent(m, c, x, y, alpha):
    dsum_m, dsum_c = 0.0, 0.0
    for i in range(len(y)):
        d_m = (m * x[i] + c - y[i]) * x[i]
        dsum_m += d_m
        d_c = (m * x[i] + c - y[i])
        dsum_c += d_c
    new_c, new_m = c - alpha * dsum_c, m - alpha * dsum_m
    return new_c, new_m

def comp_y(m1,c1,x):
    comp=[]
    for i in range(len(y)):
        comp.append(m1 * x[i] + c1)
    return comp

def fit(x, y, alpha=0.001, threshhold = 1000):
    m = np.random.random()/ 10
    c = np.random.random()/ 10
    cost_func_values = []
    for i in range(threshhold):
        computed_y = comp_y(m, c, x)
        cost_func_values.append(cost_func(computed_y, y))
        c, m = gradient_descent(m, c, x, y, alpha)
        
    return m, c