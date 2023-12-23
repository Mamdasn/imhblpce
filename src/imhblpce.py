import numpy as np
import cvxpy as cp
import numba
from imhist import imhist


def imhblpce(image):
    [h, w] = image.shape
    hist_image = imhist(image)
    PMFx = hist_image / (h * w)
    # CDFx = np.cumsum(PMFx)
    r = (PMFx > 0) * np.arange(1, 257)
    r = r[r > 0]
    r_inv = np.where(PMFx == 0, PMFx, np.ones_like(PMFx))
    r_inv = np.where(PMFx == 0, r_inv, np.cumsum(r_inv)).astype(np.int)
    # K = len(r)

    N = len(r)
    # making W
    # sigma
    sigma = 20
    # Creating Q(W, P)
    W = _Creating_W(r, sigma)
    P = _Creatng_P(PMFx, sigma, N)

    # Q is the H in the quadprog equation : minimum( 0.5 * y'*H*y )
    Q = P * W

    y = _solve_quadratic_problem(Q, N)
    # x = D\y
    # D = np.diag(np.ones((N-1))) + np.diag(-np.ones((N-2)),k=-1)
    D_inv = np.tril(np.ones((N - 1, N - 1)), k=0)

    x = np.round(np.matmul(D_inv, y))
    x = np.where(x <= 256, x, 256 * np.ones_like(x))
    x = np.where(x >= 1, x, np.ones_like(x))

    x = _padzero(x - 1).astype(np.uint8)
    img_out = _map_image(image, r_inv, x)
    # img_out = _map_image(image, r, x)
    return img_out


@numba.njit()
def _map_image(image, r_inv, x):
    output = image.copy().reshape(-1)
    for i in range(output.size):
        pixel_intensity = output[i]
        output[i] = x[r_inv[pixel_intensity] - 1]
    return output.copy().reshape((image.shape))


# def _map_image(image, r, x):
#     img_out = image.copy()
#     N = len(r)
#     for i in range(N):
#         indx = N-i-1
#         img_out[np.where(image==r[indx]-1)] = x[indx]
#     return img_out


def _padzero(arr):
    padded_arr = np.zeros((arr.size + 1), dtype=arr.dtype)
    padded_arr[1:] = arr.copy().reshape(-1)
    return padded_arr


def _solve_quadratic_problem(Q, N):
    # 1/2 y.T Q y
    # Gx ≤ h, Ax = b
    N = Q.shape[0] + 1
    A = np.ones((1, N - 1), dtype=np.float64)
    b = np.array([256.0], dtype=np.float64)
    lb = np.zeros((N - 1), dtype=np.float64)
    ub = np.ones((N - 1), dtype=np.float64) * 255
    y = cp.Variable(N - 1)
    constraints = [y >= lb, y <= ub, A @ y == b]
    prob = cp.Problem(cp.Minimize(cp.quad_form(y, Q)), constraints=constraints)
    prob.solve()
    return np.array(y.value)


def _Creating_W(r, sigma):
    N = len(r)
    # r = np.insert(r, 0, 0) # r is N
    r = _padzero(r)
    temp_1 = np.diag(np.arange(2, N + 1))
    temp_2 = np.diag(np.arange(1, N))

    w_temp = np.exp(-(r[temp_1] - r[temp_2]) / (2 * sigma**2))
    # w_temp[np.where(w_temp==1)] = 0
    w_temp = np.where(w_temp != 1, w_temp, np.zeros_like(w_temp))

    w_diag = w_temp**2

    temp_1 = np.diag(np.arange(3, N + 1), -1)
    temp_2 = np.diag(np.arange(2, N), -1)
    w_temp_1 = np.exp(-(r[temp_1] - r[temp_2]) / (2 * sigma**2))

    temp_1 = np.diag(np.arange(2, N), -1)
    temp_2 = np.diag(np.arange(1, N - 1), -1)
    w_temp_2 = np.exp(-(r[temp_1] - r[temp_2]) / (2 * sigma**2))

    w_1 = w_temp_1 * w_temp_2
    # w_1[np.where(w_1==1)] = 0
    w_1 = np.where(w_1 != 1, w_1, np.zeros_like(w_1))
    w_1 = -w_1
    w_2 = w_1.T

    W = w_diag + w_1 + w_2
    return W


def _Creatng_P(PMFx, sigma, N):
    temp_1 = np.zeros((N - 1, N - 1), dtype=np.uint16)
    temp_1[0 : N - 2, 0 : N - 2] = np.diag(np.arange(3, N + 1))
    temp_2 = np.zeros((N - 1, N - 1), dtype=np.uint16)
    temp_2[1 : N - 1, 1 : N - 1] = np.diag(np.arange(2, N))
    # PMFx = np.insert(PMFx, 0, 0) # appending a zero to the left of PFMx
    PMFx = _padzero(PMFx)  # appending a zero to the left of PFMx

    PMF_diag = PMFx[temp_1] ** 2 + PMFx[temp_2] ** 2

    temp_1 = np.diag(np.arange(2, N), -1)
    temp_2 = np.diag(np.arange(3, N + 1), -1)
    PMF_temp = PMFx[temp_1] * PMFx[temp_2]

    P = PMF_diag + PMF_temp + PMF_temp.T
    return P
