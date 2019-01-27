import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_guassian_noise(im, prop, var_sigma):
    N = int(np.round(np.prod(im.shape) * prop))

    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N], im.shape)
    e = var_sigma * np.random.randn(np.prod(im.shape)).reshape(im.shape)
    im2 = np.copy(im).astype('float')
    im2[index] += e[index]

    return im2


def add_saltnpepper_noise(im, prop):
    N = int(np.round(np.prod(im.shape) * prop))
    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N], im.shape)
    im2 = np.copy(im)
    im2[index] = 1 - im2[index]

    return im2


def neighbours(i, j, M, N, size=4):
    if size == 4:
        # corners
        if i == 0 and j == 0:
            n = [(0, 1), (1, 0)]
        elif i == 0 and j == N - 1:
            n = [(0, N - 2), (1, N - 1)]
        elif i == M - 1 and j == 0:
            n = [(M - 1, 1), (M - 2, 0)]
        elif i == M - 1 and j == N - 1:
            n = [(M - 1, N - 2), (M - 2, N - 1)]

        # edges
        elif i == 0:
            n = [(0, j - 1), (0, j + 1), (1, j)]
        elif i == M - 1:
            n = [(M - 1, j - 1), (M - 1, j + 1), (M - 2, j)]
        elif j == 0:
            n = [(i - 1, 0), (i + 1, 0), (i, 1)]
        elif j == N - 1:
            n = [(i - 1, N - 1), (i + 1, N - 1), (i, N - 2)]

        # middle squares
        else:
            n = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]

        
    if size==8 :
        # corners
        if i == 0 and j == 0:
            n = [(0, 1), (1, 0), (1,1)]
        elif i == 0 and j == N - 1:
            n = [(0, N - 2), (1, N - 1), (1, N - 2)]
        elif i == M - 1 and j == 0:
            n = [(M - 1, 1), (M - 2, 0), (M - 2, 1)]
        elif i == M - 1 and j == N - 1:
            n = [(M - 1, N - 2), (M - 2, N - 1), (M - 2,N - 2)]

        # edges
        elif i == 0:
            n = [(0, j - 1), (0, j + 1), (1, j), (1, j+1), (1, j-1)]
        elif i == M - 1:
            n = [(M - 1, j - 1), (M - 1, j + 1), (M - 2, j),(M-2,j-1),(M-2,j+1)]
        elif j == 0:
            n = [(i - 1, 0), (i + 1, 0), (i, 1),(i-1,1),(i+1,1)]
        elif j == N - 1:
            n = [(i - 1, N - 1), (i + 1, N - 1), (i, N - 2),(i+1,N-2),(i+1,N-2)]

        # middle squares
        else:
            n = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1),(i-1,j-1),(i-1,j+1),(i+1,j-1),(i+1,j+1)]
    
    return n

def prior(latent_x, neighs, w):
    return sum(w * latent_x[n_j, n_i] for n_j, n_i in neighs)

def variational_bayes(noisey_image, max_iters, neighbour_size):
    ## weight of 1
    w = 1
    
    latent_x = np.copy(noisey_image) * 2 - 1
    N = latent_x.shape[0]
    M = latent_x.shape[1]
    
    for t in range(max_iters):
        for j in range(N):
            for i in range(M):
                neighs = neighbours(j, i, N, M, neighbour_size)
                
                prior_val = prior(latent_x, neighs, w)
                
                latent_x[j, i] = np.tanh(prior_val + 0.5*(latent_x[j, i] + latent_x[j, i]))
                
    return latent_x


prop = 0.5
var_sigma = 0.1

im = cv2.imread('pug.png',cv2.IMREAD_GRAYSCALE)
im = im/255.0
im[im<0.5]=-1
im[im>=0.5]=1
fig = plt.figure()
ax = fig.add_subplot(231)
ax.imshow(im, cmap='gray')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_title('original')


im2 = add_guassian_noise(im, prop, var_sigma)
ax = fig.add_subplot(232)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.imshow(im2, cmap='gray')
ax.set_title('gaussian')

im3 = add_saltnpepper_noise(im, prop)
ax = fig.add_subplot(233)
ax.imshow(im3, cmap='gray')
ax.set_title('saltnpepper')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

im_out = variational_bayes(im3,100,4)

ax = fig.add_subplot(234)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.imshow(im_out, cmap='gray')
ax.set_title('recovered')