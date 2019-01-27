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

        # everywhere else
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

        # everywhere else
        else:
            n = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1),(i-1,j-1),(i-1,j+1),(i+1,j-1),(i+1,j+1)]
    
    return n

def likelihood(j, i, latent_x, noisy_img):
    xi = (latent_x[j][i] + 1) / 2
    yi = noisy_img[j][i]
    
    # if li is 0 (xi=yi) so e^0 is 1 
    Li = - np.abs(xi - yi)
    return np.exp(Li)

def prior(j, i, latent_x, M, N, w):
    #p(x) = 1/Z0.e^E_0(x)
    
    # get the neighbours of x_ij
    neighs = neighbours(j, i, M, N, 8)
    total = 0
    for neigh_j, neigh_i in neighs:
        total += w * latent_x[j, i] * latent_x[neigh_j, neigh_i]
        
        # ... then for each neighbour of the neighbour (neighbourception), we do the same
        for neigh_jj, neigh_ii in neighbours(neigh_j, neigh_i, M, N):
            total += w * latent_x[neigh_j, neigh_i] * latent_x[neigh_jj, neigh_ii]
    
    return np.exp(total)

def posterior(j, i, label, latent_x, noisy_img, M, N):
    ## set the current position in the latent image to be the label
    latent_x[j][i] = label
    
    liklihood_prob = likelihood(j, i, latent_x, noisy_img)
    
    # calculate prior , keep a 70% belief on the prior assumptions
    w = 0.7
    prior_prob = prior(j, i, latent_x, M, N, w)
    
    return liklihood_prob * prior_prob
    

def gibbs_sampler_denoise(noisy_img, time_steps, burn_in):
    latent_x = np.copy(noisy_img)
    
    # for time_steps iterations
    estimate = np.zeros_like(latent_x).astype(np.float64)
    for t in range(time_steps + burn_in):
        print(t)
        for j in range(noisy_img.shape[0]):
            for i in range(noisy_img.shape[1]):

                post_x1 = posterior(j, i, 1, latent_x, noisy_img, noisy_img.shape[0], noisy_img.shape[1])
                post_notx1 = posterior(j, i, -1, latent_x, noisy_img, noisy_img.shape[0], noisy_img.shape[1])
                
                ## calculate gibbs post
                gibbs_post = post_x1 / (post_notx1 + post_x1)
                z = np.random.uniform(0, 1)
                if gibbs_post > z:
                    latent_x[j, i] = 1
                else:
                    latent_x[j, i] = -1
                
                # add to estimate if we've finished burning in
                if t > burn_in:
                    estimate += latent_x
    return estimate / time_steps

prop = 0.3
var_sigma = 1

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

np.random.seed(42)
im_out = gibbs_sampler_denoise(im3, 0, 2)
im_out[im_out >= 0] = 1
im_out[im_out < 0]  = -1

fig = plt.figure()

## uncomment to show triple plot 

# ax = fig.add_subplot(131)
# ax.imshow(im, cmap='gray')

# ax2 = fig.add_subplot(132)
# ax2.imshow(im3, cmap='gray')

ax3 = fig.add_subplot(133)
ax3.imshow(im_out, cmap='gray')
plt.show()