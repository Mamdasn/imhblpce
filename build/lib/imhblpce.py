import numpy as np
import cvxpy as cp

def imhist(image):
    hist, _ = np.histogram(image.reshape(1, -1), bins=256, range=(0, 255))
    return hist
    
def imhblpce(image):
    [h, w] = image.shape
    hist_image = imhist(image)
    PMFx = hist_image / (h*w)
    # CDFx = np.cumsum(PMFx)
    r = (PMFx>0) * [i for i in range(1, 257)]
    r = r[r>0]
    # K = len(r)

    N = len(r)
    # making W
    #sigma
    sigma = 20
    # Creating Q(W, P)
    W = _Creating_W(r, sigma)
    P = _Creatng_P(PMFx, sigma, N)
    # Q is the H in the quadprog equation : minimum( 0.5 * y'*H*y )
    Q = P * W
    
    def solve_quadratic_problem(Q, N):
        # 1/2 y.T Q y
        # Gx ≤ h, Ax = b
        N = Q.shape[0] + 1
        A = np.ones((1, N-1), dtype=np.float)
        b = np.array([256.], dtype=np.float)
        lb = np.zeros((N-1), dtype=np.float)
        ub =  np.ones((N-1), dtype=np.float) * 255
        y = cp.Variable(N-1)
        constraints = [y >= lb, y <= ub, A @ y == b]
        prob = cp.Problem(
                    cp.Minimize(cp.quad_form(y, Q)),
                    constraints=constraints
                    )

        prob.solve()
        return np.array(y.value)
    y = solve_quadratic_problem(Q, N)
    
    # x = D\y
    # D = np.diag(np.ones((N-1))) + np.diag(-np.ones((N-2)),k=-1)
    D_inv = np.tril(np.ones((N-1, N-1)), k=0)
    
    x = np.round(np.matmul(D_inv, y))
    x = np.where(x<=256, x, 256*np.ones_like(x))
    x = np.where(x>=1, x, np.ones_like(x))
    
    xx = np.ones((N))
    xx[1:N] = x
    xx = xx - 1
    xx = xx.astype(np.uint8)

    img_out = image.copy()
    for i in range(N):
        indx = N-i-1
        img_out[np.where(image==r[indx]-1)] = xx[indx]
    return img_out

def _Creating_W(r, sigma): 
    N = len(r)
    r = np.insert(r, 0, 0) # r is N

    temp_1 = np.diag(([i for i in range(2, N+1)]))
    temp_2 = np.diag(([i for i in range(1, N)]))

    w_temp = np.exp(-(r[temp_1]-r[temp_2])/(2*sigma**2))
    w_temp[np.where(w_temp==1)] = 0

    w_diag = w_temp**2

    temp_1 = np.diag([i for i in range(3, N+1)], -1)
    temp_2 = np.diag([i for i in range(2, N)], -1)
    w_temp_1 = np.exp(-(r[temp_1]-r[temp_2])/(2*sigma**2))

    temp_1 = np.diag([i for i in range(2, N)], -1)
    temp_2 = np.diag([i for i in range(1, N-1)], -1)
    w_temp_2 = np.exp(-(r[temp_1]-r[temp_2])/(2*sigma**2))

    w_1 = w_temp_1 * w_temp_2
    w_1[np.where(w_1==1)] = 0
    w_1 = -w_1
    w_2 = w_1.T

    W = w_diag + w_1 + w_2
    return W

def _Creatng_P(PMFx, sigma, N):

    temp_1 = np.zeros((N-1,N-1), dtype=np.uint16)    
    temp_1[0:N-2,0:N-2] = np.diag([i for i in range(3, N+1)])
    temp_2 = np.zeros((N-1,N-1), dtype=np.uint16)
    temp_2[1:N-1,1:N-1] = np.diag([i for i in range(2, N)])
    PMFx = np.insert(PMFx, 0, 0) # appending a zero to the left of PFMx

    PMF_diag = PMFx[temp_1]**2 + PMFx[temp_2]**2

    temp_1 = np.diag([i for i in range(2, N)], -1)
    temp_2 = np.diag([i for i in range(3, N+1)], -1)
    PMF_temp = PMFx[temp_1] * PMFx[temp_2]

    P = PMF_diag + PMF_temp + PMF_temp.T
    return P
