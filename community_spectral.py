# -*- coding: utf-8 -*-
"""
Algorithms for Spectral Clustering based Spatial Community Detection(SCSCD).
"""

# Author: You Wan <wanyou9@gmail.com>
# Data: 2021.10.13
# License: BSD 3 clause


__all__ = ['spectral_clustering_adp']

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.manifold import spectral_embedding
from .region_k_means import region_k_means
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse.linalg import eigsh
from scipy import sparse
def _set_diag(laplacian, value, norm_laplacian):
    """Set the diagonal of the laplacian matrix and convert it to a
    sparse format well suited for eigenvalue decomposition.

    Parameters
    ----------
    laplacian : {ndarray, sparse matrix}
        The graph laplacian.

    value : float
        The value of the diagonal.

    norm_laplacian : bool
        Whether the value of the diagonal should be changed or not.

    Returns
    -------
    laplacian : {array, sparse matrix}
        An array of matrix in a form that is well suited to fast
        eigenvalue decomposition, depending on the band width of the
        matrix.
    """
    n_nodes = laplacian.shape[0]
    # We need all entries in the diagonal to values
    if not sparse.isspmatrix(laplacian):
        if norm_laplacian:
            laplacian.flat[::n_nodes + 1] = value
    else:
        laplacian = laplacian.tocoo()
        if norm_laplacian:
            diag_idx = (laplacian.row == laplacian.col)
            laplacian.data[diag_idx] = value
        # If the matrix has a small number of diagonals (as in the
        # case of structured matrices coming from images), the
        # dia format might be best suited for matvec products:
        n_diags = np.unique(laplacian.row - laplacian.col).size
        if n_diags <= 7:
            # 3 or less outer diagonals on each side
            laplacian = laplacian.todia()
        else:
            # csr has the fastest matvec and is thus best suited to
            # arpack
            laplacian = laplacian.tocsr()
    return laplacian


def predict_k(affinity_matrix, nc):
    """
    Predict number of clusters based on the eigengap.

    Parameters
    ----------
    affinity_matrix : array-like or sparse matrix, shape: (n_samples, n_samples)
        adjacency matrix.
        Each element of this matrix contains a measure of similarity between two of the data points.

    Returns
    ----------
    k : integer
        estimated number of cluster.

    Note
    ---------
    If graph is not fully connected, zero component as single cluster.

    References
    ----------
    A Tutorial on Spectral Clustering, 2007
        Luxburg, Ulrike
        http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf

    """

    """
    If normed=True, L = D^(-1/2) * (D - A) * D^(-1/2) else L = D - A.
    normed=True is recommended.
    """
    normed_laplacian, dd = csgraph_laplacian(affinity_matrix, normed=True, return_diag=True)
    laplacian = _set_diag(normed_laplacian, 1,True)
    laplacian *= -1
    """
    n_components size is N - 1.
    Setting N - 1 may lead to slow execution time...
    """
    n_components = nc # affinity_matrix.shape[0] - 1

    """
    shift-invert mode
    The shift-invert mode provides more than just a fast way to obtain a few small eigenvalues.
    http://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html

    The normalized Laplacian has eigenvalues between 0 and 2.
    I - L has eigenvalues between -1 and 1.
    """
    # v0 = _init_arpack_v0(laplacian.shape[0], random_state)
    eigenvalues, eigenvectors = eigsh(laplacian, k=n_components, which="LM", sigma=1.0, maxiter=5000)
    eigenvalues = -eigenvalues[::-1]  # Reverse and sign inversion.
    #
    #
    # max_gap = 0
    # gap_pre_index = 0
    # for i in range(1, eigenvalues.size):
    #     gap = eigenvalues[i] - eigenvalues[i-1]
    #     if gap > max_gap and i<int(n_components/5):
    #         max_gap = gap
    #         gap_pre_index = i - 1

    '''
    ev = list(eigenvalues)
    ev.sort(reverse=False)
    idiff = [ev[i] - ev[i-1] for i in range(1,len(ev))]
    imax = np.where(idiff == max(idiff))[0][0]
    # while imax>int(n_components/10):
    #     idiff[imax] = min(idiff)
    #     imax = np.where(idiff == max(idiff))[0][0]
    rid = list(range(20))     # len(idiff)
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_xlim(0,max(rid))
    ax1.set_ylim(0,max(idiff))
    plt.plot(rid,idiff[:20])
    plt.plot(imax,max(idiff),'ro')
    plt.show()
    '''
    # k = gap_pre_index + 1

    return eigenvalues

def spectral_clustering_adp(cg, nc, w, aff='precomputed'):

    # affinity_matrix_ = np.array(nx.adjacency_matrix(cg).todense())
    cg_size=len(cg.nodes())
    am = np.zeros([cg_size,cg_size])
    for inode in cg.nodes():
        for jnode in cg.adj[inode]:
            am[inode][jnode] = cg[inode][jnode]['weight']
        # am[inode][inode] = sum(am[inode])
        # am[inode] = am[inode] / am[inode][inode]

    affinity_matrix_ = am

    eigenvalues  = predict_k(affinity_matrix_,nc)
    y_pred = SpectralClustering(n_clusters=nc, affinity=aff,nb_matrix=w).fit(affinity_matrix_) #, eigen_solver='arpack', eigen_tol=50, floor_weights=np.array(data), floor = y_floor_weight

    y_lab = y_pred.labels_
    a2r = dict()
    for i in range(len(y_lab)):
        a2r[i] = y_lab[i]

    return a2r, eigenvalues


class SpectralClustering(ClusterMixin, BaseEstimator):
    """Apply clustering to a projection of the normalized Laplacian.

    In practice Spectral Clustering is very useful when the structure of
    the individual clusters is highly non-convex or more generally when
    a measure of the center and spread of the cluster is not a suitable
    description of the complete cluster. For instance when clusters are
    nested circles on the 2D plane.

    If affinity is the adjacency matrix of a graph, this method can be
    used to find normalized graph cuts.

    When calling ``fit``, an affinity matrix is constructed using either
    kernel function such the Gaussian (aka RBF) kernel of the euclidean
    distanced ``d(X, X)``::

            np.exp(-gamma * d(X,X) ** 2)

    or a k-nearest neighbors connectivity matrix.

    Alternatively, using ``precomputed``, a user-provided affinity
    matrix can be used.

    Read more in the :ref:`User Guide <spectral_clustering>`.

    Parameters
    ----------
    n_clusters : int, default=8
        The dimension of the projection subspace.

    eigen_solver : {'arpack', 'lobpcg', 'amg'}, default=None
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems,
        but may also lead to instabilities. If None, then ``'arpack'`` is
        used.

    n_components : int, default=n_clusters
        Number of eigen vectors to use for the spectral embedding

    random_state : int, RandomState instance, default=None
        A pseudo random number generator used for the initialization of the
        lobpcg eigen vectors decomposition when ``eigen_solver='amg'`` and by
        the K-Means initialization. Use an int to make the randomness
        deterministic.
        See :term:`Glossary <random_state>`.

    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    gamma : float, default=1.0
        Kernel coefficient for rbf, poly, sigmoid, laplacian and chi2 kernels.
        Ignored for ``affinity='nearest_neighbors'``.

    affinity : str or callable, default='rbf'
        How to construct the affinity matrix.
         - 'nearest_neighbors' : construct the affinity matrix by computing a
           graph of nearest neighbors.
         - 'rbf' : construct the affinity matrix using a radial basis function
           (RBF) kernel.
         - 'precomputed' : interpret ``X`` as a precomputed affinity matrix.
         - 'precomputed_nearest_neighbors' : interpret ``X`` as a sparse graph
           of precomputed nearest neighbors, and constructs the affinity matrix
           by selecting the ``n_neighbors`` nearest neighbors.
         - one of the kernels supported by
           :func:`~sklearn.metrics.pairwise_kernels`.

        Only kernels that produce similarity scores (non-negative values that
        increase with similarity) should be used. This property is not checked
        by the clustering algorithm.

    n_neighbors : int
        Number of neighbors to use when constructing the affinity matrix using
        the nearest neighbors method. Ignored for ``affinity='rbf'``.

    eigen_tol : float, default=0.0
        Stopping criterion for eigendecomposition of the Laplacian matrix
        when ``eigen_solver='arpack'``.

    assign_labels : {'kmeans', 'discretize'}, default='kmeans'
        The strategy to use to assign labels in the embedding
        space. There are two ways to assign labels after the laplacian
        embedding. k-means can be applied and is a popular choice. But it can
        also be sensitive to initialization. Discretization is another approach
        which is less sensitive to random initialization.

    degree : float, default=3
        Degree of the polynomial kernel. Ignored by other kernels.

    coef0 : float, default=1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : dict of str to any, default=None
        Parameters (keyword arguments) and values for kernel passed as
        callable object. Ignored by other kernels.

    n_jobs : int, default=None
        The number of parallel jobs to run when `affinity='nearest_neighbors'`
        or `affinity='precomputed_nearest_neighbors'`. The neighbors search
        will be done in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : bool, default=False
        Verbosity mode.

        .. versionadded:: 0.24

    Attributes
    ----------
    affinity_matrix_ : array-like of shape (n_samples, n_samples)
        Affinity matrix used for clustering. Available only if after calling
        ``fit``.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point

    Examples
    --------
    >>> from sklearn.cluster import SpectralClustering
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [1, 0],
    ...               [4, 7], [3, 5], [3, 6]])
    >>> clustering = SpectralClustering(n_clusters=2,
    ...         assign_labels="discretize",
    ...         random_state=0).fit(X)
    >>> clustering.labels_
    array([1, 1, 1, 0, 0, 0])
    >>> clustering
    SpectralClustering(assign_labels='discretize', n_clusters=2,
        random_state=0)

    Notes
    -----
    If you have an affinity matrix, such as a distance matrix,
    for which 0 means identical elements, and high values means
    very dissimilar elements, it can be transformed in a
    similarity matrix that is well suited for the algorithm by
    applying the Gaussian (RBF, heat) kernel::

        np.exp(- dist_matrix ** 2 / (2. * delta ** 2))

    Where ``delta`` is a free parameter representing the width of the Gaussian
    kernel.

    Another alternative is to take a symmetric version of the k
    nearest neighbors connectivity matrix of the points.

    If the pyamg package is installed, it is used: this greatly
    speeds up computation.

    References
    ----------

    - Normalized cuts and image segmentation, 2000
      Jianbo Shi, Jitendra Malik
      http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.160.2324

    - A Tutorial on Spectral Clustering, 2007
      Ulrike von Luxburg
      http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323

    - Multiclass spectral clustering, 2003
      Stella X. Yu, Jianbo Shi
      https://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf
      
     - Spatially–encouraged spectral clustering: a technique for blending map typologies and regionalization, 2021
      Levi John Wolf
      https://www.tandfonline.com/doi/full/10.1080/13658816.2021.1934475
      https://osf.io/fcs5x/
    """
    @_deprecate_positional_args
    def __init__(self, n_clusters=8, *, eigen_solver=None, nb_matrix=None,
                 n_init=10, gamma=1., affinity='rbf',
                 n_neighbors=10, eigen_tol=1e-9, assign_labels='kmeans',
                 degree=3, coef0=1, kernel_params=None, n_jobs=1):
        self.n_clusters = n_clusters
        self.eigen_solver = eigen_solver
        self.n_components = n_clusters
        self.nb_matrix = nb_matrix
        self.random_state =None
        self.n_init = n_init
        self.gamma = gamma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.eigen_tol = eigen_tol
        self.assign_labels = assign_labels
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs
        self.verbose = False
        self.sse = []

    def top_seed(self, X):
        k = self.n_clusters
        w = self.nb_matrix
        nb_w = w.weights
        degree_X = sum(X)
        degree_X_argSort = np.argsort(degree_X)
        n_clust2 = degree_X_argSort[-k*k:len(X)]
        top_s = True
        while top_s:
            nb_clust = []
            n_clust = np.random.choice(n_clust2, size=k, replace=False)
            print("n_clust_seeds: ",n_clust)
            for i in n_clust:
                nb_clust.extend(w[i])
            if len(set(nb_clust).intersection(set(n_clust)))==0:
                top_s = False
        return n_clust

    def fit(self, X, y=None):
        """Perform spectral clustering from features, or affinity matrix.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
                array-like of shape (n_samples, n_samples)
            Training instances to cluster, or similarities / affinities between
            instances if ``affinity='precomputed'``. If a sparse matrix is
            provided in a format other than ``csr_matrix``, ``csc_matrix``,
            or ``coo_matrix``, it will be converted into a sparse
            ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self

        """
        self.affinity_matrix_ = X
        n_clust = self.top_seed(X)
        maps = spectral_embedding(self.affinity_matrix_, n_components=self.n_components, drop_first=False)


        centroid, self.labels_, inits, self.sse = region_k_means(X=maps, n_clusters=n_clust, w=self.nb_matrix)

        return self
