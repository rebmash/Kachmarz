import numpy as np
from numpy import linalg as LA

def stepsize_func(g, A,  a, B, x0):
    q = x0.dot(A.dot(x0)) * 0.5
    if abs(g.dot(A.dot(g))) < 1e-8: 
        return (a - q) / (g.dot(A.dot(x0)))
        #print("aaa")
        #return 2 * (a - q)
    return ((-g.dot(A.dot(x0))) + (max(g.dot(B.dot(g)), 0)**0.5)) / (g.dot(A.dot(g)))
    #return ((-g.dot(A.dot(x0))) + (g.dot(B.dot(g))**0.5)) / (g.dot(A.dot(g)))
    
    
def proj(A, a, x0):
    n = len(A)
    eps = 1e-10
    max_iter = 1e3
    B = np.outer(A.dot(x0), (A.dot(x0))) + 2 * (a - 0.5 * (x0.dot(A.dot(x0)))) * A
    w, v = LA.eig(B)
    #print(w)
    ind = -1
    
    ans = float("-inf")
    for i in range(len(w)):
        if w[i] > ans:
            ans = w[i]
            ind = i
    d = v[:,ind].copy()
    if ans < 1e-11:
        return [None] * n, []
    elif abs(x0.dot(A.dot(x0)) - a) < eps:
        return x0, [0], [x0]
    else:
        if (d.dot(A.dot(x0))) < 0:
            d = -d
        num_of_iter = 0
        t = stepsize_func(d, A, a, B, x0)
        converge = []
        points = []
        x = x0 + t * d
        while abs(d.dot(A.dot(x)) - LA.norm(A.dot(x))) > eps and num_of_iter < max_iter:
            num_of_iter += 1
            m = 1
            flag = 0
            kek = 0
            while flag == 0:
                kek += 1
                d_a = (d + ((1/2)**m) * (A.dot(x) - d))
                d_a = (d_a / LA.norm(d_a))

                if (d_a.dot(A.dot(x0))) < 0:
                    d_a = (-d_a)



                d_c = (2 * (d.dot(A.dot(x))) * d - A.dot(x))
                d_b = (d + ((1/2)**m) * (d_c - d))
                d_b = ((d_b) / LA.norm(d_b))

                if (d_b.dot(A.dot(x0))) < 0:
                    d_b = (-d_b)


                if d_a.dot(B.dot(d_a)) >= 0:
                    t_a = stepsize_func(d_a, A, a, B, x0)
                    if abs(t_a) < abs(t):
                        flag = 1


                if d_b.dot(B.dot(d_b)) >= 0 :
                    t_b = stepsize_func(d_b, A, a, B, x0)
                    if flag == 1 and abs(t_b) < abs(t_a):
                        flag = 2
                    elif flag == 0 and abs(t_b) < abs(t):
                        flag = 2
                if flag == 0:
                    m = m + 1
                elif flag == 1:
                    t = t_a
                    d = d_a.copy()
                else:
                    t = t_b
                    d = d_b.copy()
                if kek > 1e4:
                    d = d_a.copy()
                    break
            x = x0 + t * d
            #print(t)
            converge.append(abs(t))
            points.append(x)
        return x, converge, points

