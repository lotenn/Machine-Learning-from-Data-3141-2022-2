import numpy as np


def get_random_centroids(X, k):
    """
    Each centroid is a point in RGB space (color) in the image.
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids.
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array.
    """
    idxs = np.random.randint(X.shape[0], size=k)
    centroids = X[idxs, :]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # make sure you return a numpy array
    return np.asarray(centroids).astype(np.float)


def lp_distance(X, centroids, p=2):
    """
    Inputs:
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of
    all points in RGB space from all centroids
    """
    distances = []
    k, _ = centroids.shape
    for c in range(k):
        mk = np.row_stack([centroids[c, :]])
        distances.append((np.sum(np.abs(X - mk) ** p, axis=1)) ** (1 / p))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return np.asarray(distances).astype(np.float)


def kmeans(X, k, p, max_iter=100, init_centroids=None):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = get_random_centroids(X, k) if init_centroids is None else init_centroids
    for _ in range(max_iter):
        # step 1 - calc distance to closest centroid and assign each point to a cluster by largest probability
        distance = lp_distance(X, centroids, p)
        classes = distance.argmin(axis=0)
        classes_sizes = np.bincount(classes)

        # step 2 - recalculate centroids according to the new clusters assignment
        old_centroids = centroids.copy()
        for c in range(k):
            indices = np.where(classes == c)[0]
            centroids[c, :] = np.sum(X[indices, :], axis=0) / classes_sizes[c]

        # stop condition
        if np.array_equal(centroids, old_centroids):
            break
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes


def kmeans_pp(X, k, p, max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    original_X = X.copy()
    init_centroids = np.zeros((k, X.shape[1]))
    n_chosen = 0

    # Choose one center uniformly at random among the data points.
    X, init_centroids, n_chosen = _select_random_row(X, init_centroids, n_chosen)

    while n_chosen < k:
        # For each data point x not chosen yet, compute the distance
        # between x and the nearest center that has already been chosen.
        distance = lp_distance(X, init_centroids, p)

        distance = np.min(distance, axis=0)  # for each instance, take the minimal distance with any pre-chosen centroid
        prop = distance / np.sum(distance)  # generate proportions from the distances
        X, init_centroids, n_chosen = _select_random_row(X, init_centroids, n_chosen, prop)  # select a new centroid based on the distances proportions

    centroids, classes = kmeans(original_X, k, p, max_iter, init_centroids=init_centroids)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes


def _select_random_row(X, init_centroids, n_chosen, prop=None):
    N = X.shape[0]
    rand_row = np.random.choice(N, p=prop)
    init_centroids[n_chosen, :] = X[rand_row, :]
    X = np.delete(X, rand_row, axis=0)
    n_chosen += 1
    return X, init_centroids, n_chosen


def inertia(X, cent, labels):
    X_dist = lp_distance(X, cent).T
    N, d = X_dist.shape
    return np.sum(X_dist[np.arange(N), labels] ** 2)


def execution(algo, params):
    """
    Execute kmean algo with given params
    :param algo:
    :param params:
    :return: inertia value of the results
    """
    cent, labels = algo(**params)
    return inertia(params['X'], cent, labels)
