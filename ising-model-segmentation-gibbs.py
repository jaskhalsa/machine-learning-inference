import itertools
import random

import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


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


## this prior method gave me weeeeeeeeird results, so made a new one instead

# def prior(j, i, X, M, N, w):
#     #p(x) = 1/Z0.e^E_0(x)
    
#     # get the neighbours of x_ij
#     neighs = neighbours(j, i, M, N, 8)
    
    
#     total = 0
#     for neigh_j, neigh_i in neighs:
#         if(X[j, i] == 0):
#             total -= w * X[j, i] * X[neigh_j, neigh_i]
#         else:
#             total -= w * X[j, i] * X[neigh_j, neigh_i]
        
#         # ... then for each neighbour, we do the same
#         for neigh_jj, neigh_ii in neighbours(neigh_j, neigh_i, M, N):
#             if(X[neigh_jj, neigh_ii] == 0):
#                 total -= w * X[neigh_j, neigh_i] * X[neigh_jj, neigh_ii]
#             else:
#                 total += w * X[neigh_j, neigh_i] * X[neigh_jj, neigh_ii]
    
#     return np.exp(total)

def multiply_by_prior(j, i, fore_mask, back_mask, fLi,  bLi):
    neighs = neighbours(j, i, fore_mask.shape[0], fore_mask.shape[1])
    for neigh_j, neigh_i in neighs:
        if(fore_mask[neigh_j,neigh_i] == 1): 
            fLi *= 0.7
        else:
            fLi*= 0.3
        if(back_mask[neigh_j, neigh_i] == 1):
            bLi *= 0.7
        else:
            bLi *= 0.3
    return fLi, bLi

def get_likelihood_prob(dist, label, intensities, bins):
    idxs = enumerate(intensities)
    distProbs = []
    for (i, intensity) in idxs:
        bin_edges = bins[label, i, :]
        bin_idx = 0
        for j in range(0, len(bin_edges)):
            if (bin_edges[j] < float(intensity) and bin_edges[j+1] > float(intensity)):
                bin_idx = j
        distProbs.append(dist[label, i, bin_idx])
    return np.prod(distProbs)

    # return np.prod([dist[label, i, intensity] for (i, intensity) in idxs])

def gibbs_image_segmentation(image, burn_in, iterations, histograms, bins, fore_mask, back_mask):
    latent_x = np.zeros((image.shape[0], image.shape[1]))

    ## make random estimates for giraffe
    estimates = (np.random.random( (image.shape[0], image.shape[1]) ) > .5).astype(int)

    total_iterations = burn_in + iterations
    pixel_indices = list(itertools.product(range(image.shape[0]),range(image.shape[1])))

    for iteration in range(total_iterations):
        print(iteration)

        # Loop over entire grid, using a random order for faster convergence
        random.shuffle(pixel_indices)
        for (i,j) in pixel_indices:
            ## first we work out our likelihood for foreground and background

            xf = get_likelihood_prob(histograms, 0, image[i,j,:], bins)
            xb = get_likelihood_prob(histograms, 1, image[i,j,:], bins)
            
            postF, postB = multiply_by_prior(i, j, fore_mask, back_mask, xf, xb)
            
            pf = postF / (postF + postB)
            if(np.random.uniform(0, 1) < pf):
                estimates[i,j] = -1
            else:
                estimates[i,j] = 1
        if iteration > burn_in:
            latent_x += estimates
    
    latent_x /= total_iterations

    return latent_x

def create_histogram_dist(image, mask):
    r_hist, rbin_edges = np.histogram(image[...,0], weights=mask, bins=15, density=True)
    g_hist, gbin_edges = np.histogram(image[...,1], weights=mask, bins=15, density=True)
    b_hist, bbin_edges = np.histogram(image[...,2], weights=mask, bins=15, density=True)

    return [r_hist, g_hist, b_hist], [rbin_edges, gbin_edges, bbin_edges]

def get_dist(image, fore_mask, back_mask):
    fore_dist, fore_bins = create_histogram_dist(image, fore_mask)
    back_dist, back_bins = create_histogram_dist(image, back_mask)
    return [fore_dist, back_dist], [fore_bins, back_bins]



image_file = 'giraffe.bmp'

image = np.array(mpimg.imread(image_file))
grey = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)

fore_mask = np.zeros_like(image[:,:,0])
back_mask = np.zeros_like(image[:,:,0])

back_mask[grey >= 115] = 1
fore_mask[grey < 100] = 1

fore_mask_copy = np.zeros((image.shape[0], image.shape[1]))


back_mask_copy = np.zeros((image.shape[0], image.shape[1]))


fig, axes = plt.subplots(nrows=1, ncols=2)
axes.flat[0].imshow(fore_mask)
fig, axes = plt.subplots(nrows=1, ncols=2)
axes.flat[0].imshow(back_mask)
histograms, bins = get_dist(image, fore_mask, back_mask)
histograms = np.array(histograms)
bins = np.array(bins)

# print(np.sum(histograms, 2))
print(histograms.shape)

seg_image = gibbs_image_segmentation(image, 0, 2, histograms, bins, fore_mask, back_mask)
seg_image[seg_image >= 0] = 1
seg_image[seg_image < 0]  = -1

fig, axes = plt.subplots(nrows=1, ncols=2)
axes.flat[0].imshow(image)
plt.gray()
im = axes.flat[1].imshow(seg_image.astype('float64'), interpolation='nearest')
fig.colorbar(im, ax=axes.ravel().tolist())
plt.show()