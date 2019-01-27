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
    
beta = 2.1
eta = 1.5
h = 0

def total_energy(im_latent, im_noisy):
    E = 0.0

    for j in range(im_latent.shape[0]):
        for i in range(im_latent.shape[1]):
            E += calculate_energy(im_latent, im_noisy, j, i)

    return E

def calculate_energy(im_latent, im_noisy, x, y):
    E = h * im_latent[x, y]

    # get the neighbours of this pixel
    neighs = neighbours(x, y, im_latent.shape[0], im_latent.shape[1],8)
    E -= beta * sum(im_latent[x, y] * im_latent[neigh_x, neigh_y] for neigh_x, neigh_y in neighs)

    E -= eta * im_latent[x, y] * im_noisy[x, y]
    
    return E

def get_localised_energy(current_total, im_latent, im_noisy, j, i):

    latent_copy = np.copy(im_latent)
    current_energy = calculate_energy(im_latent, im_noisy, j, i)
    other_energy = current_total - current_energy

    ## flip pixel
    latent_copy[j, i] = latent_copy[j, i] * -1

    ## calculate the energy of this flipped pixel
    flipped_energy = calculate_energy(latent_copy, im_noisy, j, i)

    ##add this to create a total energy, with the flipped pixel instead
    flipped_total = other_energy + flipped_energy


    return current_total, flipped_total, im_latent[j,i], latent_copy[j,i]


def icm(image, max_iters):
    im_latent = np.copy(image)
    im_noisy = np.copy(image)
    im_latent[im_latent >= 0.5] = 1
    im_latent[im_latent < 0.5] = -1
    
    E_current = total_energy(im_latent, im_noisy)
    
    for t in range(max_iters):
        
        modified = False
        
        for j in range(im_latent.shape[0]):
            for i in range(im_latent.shape[1]):
                old = im_latent[j][i]
                current_energy_val, flipped_energy_val, original, flipped = get_localised_energy(E_current, im_latent, im_noisy, j, i)
                
                ## compare the total flipped energy against the current energy and choose to retain the flipped pixel accordingly
                if current_energy_val < flipped_energy_val:
                    im_latent[j][i] = original
                    E_current = current_energy_val
                else:
                    im_latent[j][i] = flipped
                    E_current = flipped_energy_val
                
                
                if old != im_latent[j][i]:
                    modified = True

        print (modified)
        print(t)
        ## if there is not modification in an iteration over an image, we have converged
        if not modified:
            print('converged')
            break

    return im_latent
    
prop = 0.4
var_sigma = 1

im = cv2.imread('pug.png',cv2.IMREAD_GRAYSCALE)
im = im/255.0
im[im<0.5]=-1
im[im>0.5]=1
fig = plt.figure()
ax = fig.add_subplot(131)
ax.imshow(im, cmap='gray')

im2 = add_guassian_noise(im, prop, var_sigma)
ax2 = fig.add_subplot(132)
ax2.imshow(im2, cmap='gray')
im3 = add_saltnpepper_noise(im, prop)
ax3 = fig.add_subplot(133)
ax3.imshow(im3, cmap='gray')

im_out = icm(im3, 13)
fig = plt.figure()
ax = fig.add_subplot(131)
ax.imshow(im, cmap='gray')

ax2 = fig.add_subplot(132)
ax2.imshow(im3, cmap='gray')

ax3 = fig.add_subplot(133)
ax3.imshow(im_out, cmap='gray')
plt.show()