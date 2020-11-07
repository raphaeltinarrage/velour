'''
velour -> geometry
(list of functions)

    - DTM
    - NormalizedLocalCovarianceMatrices
    - Lifting
    - IntersectionLineHyperplane
    - ProjectionToVector
    - VectorToProjection
'''

import numpy as np
import math
from sklearn.neighbors import KDTree
from scipy import spatial

def DTM(X,query_pts,m):
    '''
    Computes the values of the DTM (with exponent p=2) of the empirical measure 
    of a point cloud X. Requires sklearn.neighbors.KDTree to search nearest neighbors.
    
    Input:
        X (np.array): a Nxn numpy array representing N points in R^n.
        query_pts (np.array):  size kxN, rray of query points.
        m (float): parameter of the DTM in [0,1).
    
    Output: 
        DTM_result (np.array):sizea kx1, array contaning the DTM of the query points.
    
    Example:
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        Q = np.array([[0,0],[5,5]])
        DTM_values = DTM(X, Q, 0.3)
    '''
    N_tot = X.shape[0]     
    k = math.floor(m*N_tot)+1   #number of neighbors
    kdt = KDTree(X, leaf_size=30, metric='euclidean')
    NN_Dist, NN = kdt.query(query_pts, k, return_distance=True)  
    DTM_result = np.sqrt(np.sum(NN_Dist*NN_Dist,axis=1) / k)
    return(DTM_result)

def NormalizedLocalCovarianceMatrices(X, r):
    '''
    Compute the normalized covariance matrices at scale r of the set X.
    
    Input:
        X (np.array): size Nxn, representing N points in R^n.
        r (float): a positive parameter.
    
    Output: 
        NormLocCovMat (np.array): size Nx(n^2), the normalized covariance matrices.
    '''
    N = np.shape(X)[0]
    n = np.shape(X)[1]
    tree = spatial.KDTree(X)
    LocCovMat = np.zeros((n,n,N))
    NormLocCovMat = np.zeros((n,n,N))
    for i in range(N):
        x = X[i, :]
        ind = tree.query_ball_point(x, r)
        k = len(ind)    
        M = 0
        for j in range(k):
            z = X[i, :] - X[ind[j], :]
            M = M + np.outer(z,z)
        M = M/k
        LocCovMat[:,:,i] = M
    NormLocCovMat = LocCovMat/r**2
    return NormLocCovMat

def Lifting(X, r, gamma):
    '''
    Compute the set lifted set X_check associated to X. It is the set of pairs 
    (x,A), where x is a point of X, and A the associated local covariance matrix.
    
    Input:
        X (np.array): size Nxn, representing N points in R^n.
        r (float): a positive parameter.
        gamma (float): a non-negative parameter.
    
    Output: 
        X_check (np.array): size Nx(n+n^2), representing N points in R^(n+n^2).
    '''
    N = np.shape(X)[0]
    n = np.shape(X)[1]    
    NormLocCovMat = NormalizedLocalCovarianceMatrices(X,r)
    X_check = np.zeros((N, n+n**2))
    for i in range(N):
        X_check[i, 0:n] = X[i,:]
        X_check[i, n:(n+n**2)] = gamma*np.matrix.flatten(NormLocCovMat[:,:,i])
    return X_check

def IntersectionLineHyperplane(v, list_vertices):
    '''
    Gives the point of intersection between a line (spanned by v) and a hyperplane 
    (spanned by the points of list_vertices).
    Returns False if they do not intersect. If they do, return the barycentric 
    coordinates of the intersection points, and the distance between the point v 
    and the intersection point.    
    If h is a vector orthogonal to this hyperplane, and x_0 any point of this hyperplane, 
    then this intersection point is l*v, where l = <x_0,h>/<v,h>.
    Hence the distance between v and this intersection point is |1-l|*norm(v).
    In order to get the barycentric coordinates of this intersection point, we 
    first compute the linear coordinates of it in the basis given by 
    {list_vertices[i+1] - list_vertices[0], i}.

    Input: 
        v (np.array): size (1xm).
        list_vertices (list of np.array): list of length m  of (1xm) arrays. 
    
    Output:
        intersect (bool): True or False, whether they intersect.
        coord_bar (np.array): size (1xm), the barycentric coordinates.
        distance (float): distance between the point v and the intersection point.
    '''
    d = len(list_vertices)-1
    face_linspace = np.zeros((d+1,d))
        #a matrix containing a basis of the corresponding linear subspace, origin being list_vertices[0]
    for i in range(d):
        face_linspace[:, i] = list_vertices[i+1] - list_vertices[0]
    q,r = np.linalg.qr(face_linspace, mode = 'complete')
        #QR decomposition (remark : r is invertible since the vertices are affinely independant)
    h = q[:, d]
        #h is a vector orthogonal to the affine hyperplane spanned by list_vertices
    s = np.inner(v, h)
    if s==0:
        intersect = False
        coord_bar = np.NAN
        distance = np.NAN
    else:
        intersect = True
        l = np.inner(list_vertices[0], h)/s
        distance = np.abs(1 - l)    
        v_intersection = l*v
        w = v_intersection-list_vertices[0]        
        q,r = np.linalg.qr(face_linspace, mode='reduced')
            # QR decomposition (remark : r is invertible since the vertices are affinely independant)    
        coord_lin = w.dot(q)
            #coordinates of w in the orthonormal basis given by q
        coord_face = np.linalg.inv(r).dot(coord_lin)
            #coordinates of w in the basis given by face_linspace    
        coord_bar = np.append( 1-sum(coord_face),coord_face)
            #barycentric coordinates of v_intersection
        epsilon = np.finfo(float).eps*10 
            #a small value 
        coord_bar[np.abs(coord_bar) < epsilon] = 0
            #in order to identify zero values, we shrink the coordinates of coord_bar 
    return intersect, coord_bar, distance

def ProjectionToVector(A):
    '''
    Returns a unit vector on which the matrix A is a projection onto.
        
    Input: 
        mat (np.array): size (m x m), representing a matrix in M(R^m)
    
    Output: 
        v (np.array): size (1 x m), representing a vector in R^m
        
    Example:
        A = np.array([[1., 0.], [0., 0.]])
        velour.ProjectionToVector(A)
        ---> array([1., 0.])

    '''   
    u = np.linalg.svd(A)
    principal_vector = u[0][:,0]
    return principal_vector

def VectorToProjection(v):
    '''
    Returns the projection matrix on the linear subspace spanned by the vector v.
        
    Input: 
        v (np.array): size (1 x m), representing a vector in R^m.
    
    Output: 
        mat (np.array): size (m x m), representing a matrix in M(R^m).
        
    Example:
        v = np.array([1,0])
        velour.VectorToProjection(v)
        ---> array([[1., 0.],
                    [0., 0.]])
    '''   
    mat = np.outer(v,v)/np.linalg.norm(v)**2
    return mat