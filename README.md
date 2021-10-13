# SCSCD
A Spectral Clustering based Spatial Community Detection algorithm
===========
This is the python implementation of the Spectral Clustering based Spatial Community Detection algorithm (SCSCD).

SCSCD uses a spectral embedding method to find r eigenvectors of the spatial network, then uses spatial constraint k-means algorithm on the eigenvector matrix to cut the network at its minimum weighted edges and to find highly connected and spatial contiguous regions.

The implementation of SCSCD depends primarily on python packages: NumPy, scipy, networkx and scikit-learn. In addition, geopandas is recommended to read and write the shapefile data.
