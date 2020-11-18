'''
velour -> datasets
(list of functions)

Generate datasets:
    - SampleOnCircle
    - SampleOnOlympics
    - SampleOnLemniscate
    - SampleOnSphere
    - SampleOnNecklace
    - SampleOnCircleNormalBundle
    - SampleOnCircleMobiusBundle
    - SampleOnTorusNormalBundle
    - SampleOnKleinBottleNormalBundle

Plotting utilities:
    - COLORS
    - set_axes_equal
    - PlotPointCloud
    - PlotPCA
    - PlotVectorField
    - PlotPersistenceDiagram
    - PlotPersistenceBarcodes
    - PlotLifebar
'''

import gudhi
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from .geometry import VectorToProjection, ProjectionToVector

def SampleOnCircle(N_observation = 100, N_anomalous = 0):
    '''
    Sample N_observation points from the uniform distribution on the unit circle 
    in R^2, and N_anomalous points from the uniform distribution on the unit square.
        
    Input: 
        N_observation (int): number of sample points on the circle.
        N_noise (int, optional): number of sample points on the square.
    
    Output : 
        data (np.array): size (N_observation + N_anomalous)x2, the points concatenated.
    '''
    rand_uniform = np.random.rand(N_observation)*2-1    
    X_observation = np.cos(2*np.pi*rand_uniform)
    Y_observation = np.sin(2*np.pi*rand_uniform)
    X_anomalous = np.random.rand(N_anomalous)*2-1
    Y_anomalous = np.random.rand(N_anomalous)*2-1
    X = np.concatenate((X_observation, X_anomalous))
    Y = np.concatenate((Y_observation, Y_anomalous))
    data = np.stack((X,Y)).transpose()
    return data

def SampleOnOlympics(N_observation = 50, N_anomalous = 0):
    '''
    Sample 5*N_observation points from the uniform distribution on five unit 
    circles in R^2, and N_anomalous points from the uniform distribution on
    a rectangle.
        
    Input: 
        N_observation (int): number of sample points per circle.
        N_anomalous (int): number of sample points in rectangle.
    
    Output : 
        data (np.array): size (5*N_observation+N_anomalous)x2, the sample points. 
    '''
    l=1.25
    center1 = np.array([0,1])
    center2 = np.array([1*l,0])
    center3 = np.array([2*l,1])
    center4 = np.array([3*l,0])
    center5 = np.array([4*l,1])
    I = np.linspace(0, 2*np.pi, N_observation) 
     
    circle = np.stack([np.cos(I),np.sin(I)])    
    X1 = np.transpose(np.transpose(circle)+center1)    
    X2 = np.transpose(np.transpose(circle)+center2)    
    X3 = np.transpose(np.transpose(circle)+center3)    
    X4 = np.transpose(np.transpose(circle)+center4)    
    X5 = np.transpose(np.transpose(circle)+center5)    
    XX = np.concatenate([X1,X2,X3,X4,X5], 1)
    XX = XX.T
    
    X_out = np.random.rand(N_anomalous)*(7*l-0.25)-1.75
    Y_out = np.random.rand(N_anomalous)*4-1.5
    YY = np.stack((X_out, Y_out)).T
    data = np.vstack((XX,YY)) 
    return data

def SampleOnLemniscate(N_observation = 100, N_anomalous = 50):
    '''
    Sample N_observation evenly spaced points on Bernouilli's lemniscate, and 
    N_anomalous points from the uniform distribution on a rectangle.
    
    Input: 
        N_observation (int): number of sample points per circle.
        N_anomalous (int): number of sample points in rectangle.
    
    Output : 
        data (np.array): size (N_observation+N_anomalous)x2, the sample points. 
    '''
    I = np.linspace(0, 2*np.pi, N_observation+1)
    I = I[0:-1]
    pas = 2*np.pi/N_observation
    
    a = np.sin(I)
    b = np.cos(I)*np.sin(I)
    c = (1+np.cos(I)**2)
    ap = np.cos(I)
    bp = np.cos(I)**2-np.sin(I)**2
    cp = -2*np.cos(I)*np.sin(I)
    Xp = (a*cp-ap*c)/(c**2)
    Yp = (b*cp-bp*c)/(c**2)
    
    g = np.sqrt(  ( Xp )**2 + ( Yp )**2)
    C=np.cumsum(g)
    G = np.concatenate(([0], C[0:-1]))*pas
    alpha = G[-1]
    II = np.linspace(0, alpha, N_observation)
    Ginv = np.interp(II, G, I)
    
    X_obs=np.sin(Ginv)/(1+np.cos(Ginv)**2)
    Y_obs=np.cos(Ginv)*np.sin(Ginv)/(1+np.cos(Ginv)**2)

    X_out = np.random.rand(N_anomalous)*2-1
    Y_out = np.random.rand(N_anomalous)-0.5
    X = np.concatenate((X_obs, X_out))
    Y = np.concatenate((Y_obs, Y_out))
    data = np.stack((X,Y)).transpose() 
    
    return data

def SampleOnSphere(N_observation = 100, N_anomalous = 0):
    '''
    Sample N_observation points from the uniform distribution on the unit sphere 
    in R^3, and N_anomalous points from the uniform distribution on the unit cube.
        
    Input: 
        N_observation (int): number of sample points on the sphere.
        N_anomalous (int): number of sample points on the cube.
    
    Output: 
        data (np.array): size (N_observation + N_anomalous)x3, the points concatenated. 
        
    Example:
        X = SampleOnSphere(N_observation = 100, N_anomalous = 0)
        velour.PlotPointCloud(X)
    '''
    RAND_obs = np.random.normal(0, 1,  (3, N_observation))
    norms = np.sum(np.multiply(RAND_obs, RAND_obs).T, 1).T
    X_obs = RAND_obs[0,:]/np.sqrt(norms)
    Y_obs = RAND_obs[1,:]/np.sqrt(norms)
    Z_obs = RAND_obs[2,:]/np.sqrt(norms)    
    X_out = np.random.rand(N_anomalous)*2-1
    Y_out = np.random.rand(N_anomalous)*2-1
    Z_out = np.random.rand(N_anomalous)*2-1
    X = np.concatenate((X_obs, X_out))
    Y = np.concatenate((Y_obs, Y_out))
    Z = np.concatenate((Z_obs, Z_out))
    data = np.stack((X,Y,Z)).transpose()
    return data

def SampleOnNecklace(N_observation = 100, N_anomalous = 0):
    '''
    Sample 4*N_observation points on a necklace in R^3, and N_anomalous points 
    from the uniform distribution on a cube.
    
    Input : 
        N_observation (int): number of sample points on the sphere.
        N_anomalous (int): number of sample points on the cube.
    
    Output : 
        data (np.array): size (4*N_observation + N_anomalous)x3, the points concatenated .
    '''    
    X1 = SampleOnSphere(N_observation, N_anomalous = 0)+[2,0,0]
    X2 = SampleOnSphere(N_observation, N_anomalous = 0)+[-1,2*.866,0]
    X3 = SampleOnSphere(N_observation, N_anomalous = 0)+[-1,-2*.866,0]
    X4 = 2*SampleOnCircle(N_observation, N_anomalous = 0)
    X4 = np.stack((X4[:,0],X4[:,1],np.zeros(N_observation))).transpose()
    data_obs = np.concatenate((X1, X2, X3, X4))   
    X_out = 3*(np.random.rand(N_anomalous)*2-1)
    Y_out = 3*(np.random.rand(N_anomalous)*2-1)
    Z_out = 3*(np.random.rand(N_anomalous)*2-1)
    data_out =np.stack((X_out,Y_out,Z_out)).transpose()
    data = np.concatenate((data_obs, data_out))
    return data

def SampleOnCircleNormalBundle(N = 100, gamma=1, sd = 0):
    '''
    Returns a uniform N-sample of the normal bundle of the circle, with underlying 
    norm with parameter gamma.
    If sd is nonzero, returns a noisy sample.
    
    Input:
        N (int): number of points to sample.
        gamma (float): parameter of the underlying norm (typically gamma=1).
        sd (float): standard deviation of the noise.
    
    Output:
        X_check: a (N x (2+2**2)) np.array, representing the vector bundle.
        
    Example:
        velour.SampleOnCircleNormalCheck(N = 2, gamma=1, sd=0)
        ---> array([[ 1.00000000e+00,  0.00000000e+00,  1.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                    [ 1.00000000e+00, -2.44929360e-16,  1.00000000e+00, -2.44929360e-16, -2.44929360e-16,  5.99903913e-32]])
    '''
    U = np.linspace(0, 2*np.pi, N)
    Xdata = np.cos(U)
    Ydata = np.sin(U)
    XXnormal = gamma*np.cos(U)**2
    YYnormal = gamma*np.sin(U)**2
    XYnormal = gamma*np.cos(U)*np.sin(U)
    X_check = np.vstack((Xdata,Ydata, XXnormal, XYnormal, XYnormal, YYnormal)).T
    if sd:
        X_check = X_check + np.random.normal(loc = 0, scale = sd, size = np.shape(X_check))    
    return X_check

def SampleOnCircleMobiusBundle(N = 100, gamma=1, sd = 0):
    '''
    Returns a uniform N-sample of the normal bundle of the circle, with underlying 
    norm with parameter gamma.
    If sd is nonzero, returns a noisy sample.
    
    Input:
        N (int): number of points to sample.
        gamma (float): parameter of the underlying norm (typically gamma=1).
        sd (float): standard deviation of the noise.
    
    Output:
        X_check: a (N x (2+2**2)) np.array, representing the vector bundle.
    '''
    U = np.linspace(0, 2*np.pi, N+1)[:-1]
    Xdata = np.cos(U)
    Ydata = np.sin(U)
    XXnormal = gamma*np.cos(U/2)**2
    YYnormal = gamma*np.sin(U/2)**2
    XYnormal = gamma*np.cos(U/2)*np.sin(U/2)    
    X_check = np.vstack((Xdata,Ydata, XXnormal, XYnormal, XYnormal, YYnormal)).T
    if sd:
        X_check = X_check + np.random.normal(loc = 0, scale = sd, size = np.shape(X_check))    
    return X_check

def SampleOnTorusNormalBundle(N = 100, min_dist = 0.5, gamma=1):
    '''
    Returns a uniform sample of the normal bundle of the torus, seen in R^3, 
    with underlying norm with parameter gamma. The initial number of points is N^2, 
    and the sample is then sparsified with minimal distance between points being min_dist. 
    
    Input:
        N (int): square root of the initial number of points to sample.
        min_dist (float): final minimal distance between points.
        gamma (floatà: parameter of the underlying norm (typically gamma=1).
    
    Output:
        X_check (np.array): size (N x (3+3**2)), representing the vector bundle.
        N_points (int): the number of final points.
    '''
    U = np.linspace(0, 2*np.pi, N)
    V = np.linspace(0, 2*np.pi, N)
    Xdata = np.zeros((N,N))
    Ydata = np.zeros((N,N))
    Zdata = np.zeros((N,N))
    X_normal = []
    
    for i in range(N):
        for j in range(N):
            u = U[i]
            v = V[j]
            x = (2+np.cos(u))*np.cos(v)
            y = (2+np.cos(u))*np.sin(v)
            z = np.sin(u)
            Xdata[i,j] = x
            Ydata[i,j] = y
            Zdata[i,j] = z
            normal = np.array([np.cos(v)*np.cos(u), np.sin(v)*np.cos(u), np.sin(u)])
            X_normal.append(normal)   
    Xf = np.matrix.flatten(Xdata)
    Yf = np.matrix.flatten(Ydata)
    Zf = np.matrix.flatten(Zdata)
    X = np.vstack((Xf,Yf,Zf)).T
    
    X_map = {}
    for i in range(N**2):
        v = X_normal[i]
        A = VectorToProjection(v)
        X_map[i] = A               
    X_check = np.zeros((N**2, 3+3*3))
    for i in range(N**2):
        X_check[i, 0:3] = X[i,:]
        X_check[i, 3:12] = np.reshape(gamma*X_map[i], (1,9))    
    
    X_check = gudhi.subsampling.sparsify_point_set(points = X_check, min_squared_dist = min_dist**2)
    X_check = np.array(X_check)
    N_points = X_check.shape[0]
    print('The sample contains '+repr(N_points)+ ' points.', flush=True)
    
    return X_check, N_points

def SampleOnKleinBottleNormalBundle(N = 100, min_dist = 0.5, gamma=1):
    '''
    Returns a uniform sample of the normal bundle of the Klein bottle, seen in R^3, 
    with underlying norm with parameter gamma. The initial number of points is N^2, 
    and the sample is then sparsified with minimal distance between points being min_dist.
    
    Input:
        N (int): square root of the initial number of points to sample.
        min_dist (float): final minimal distance between points.
        gamma (float): parameter of the underlying norm (typicaly gamma=1).
    
    Output:
        X_check (np.array): size (N x (3+3**2)), representing the vector bundle.
        N_points (int): the number of final points.
    '''
    # Parameters of the Klein bottle
    a = 3
    b = 1 

    U = np.linspace(0, 2*np.pi, N)
    V = np.linspace(0, 2*np.pi, N)
    Xdata = np.zeros((N,N))
    Ydata = np.zeros((N,N))
    Zdata = np.zeros((N,N))
    X_normal = []    
    for i in range(N):
        for j in range(N):
            u = U[i]
            v = V[j]
            x = (a + b*(np.cos(u/2)*np.sin(v) - np.sin(u/2)*np.sin(2*v)))*np.cos(u)
            y = (a + b*(np.cos(u/2)*np.sin(v) - np.sin(u/2)*np.sin(2*v)))*np.sin(u)
            z = b*(np.sin(u/2)*np.sin(v) + np.cos(u/2)*np.sin(2*v))
            Xdata[i,j] = x
            Ydata[i,j] = y
            Zdata[i,j] = z
            #tangent vectors
            theta = u
            nu = v
            nx = np.array([ -1/2*(2*np.cos(nu)*np.cos(1/2*theta)*np.sin(nu) + b*np.sin(nu)*np.sin(1/2*theta))*np.cos(theta) - (b*np.cos(1/2*theta)*np.sin(nu) - 2*np.cos(nu)*np.sin(nu)*np.sin(1/2*theta) + a)*np.sin(theta), 
                             (b*np.cos(1/2*theta)*np.sin(nu) - 2*np.cos(nu)*np.sin(nu)*np.sin(1/2*theta) + a)*np.cos(theta) - 1/2*(2*np.cos(nu)*np.cos(1/2*theta)*np.sin(nu) + b*np.sin(nu)*np.sin(1/2*theta))*np.sin(theta), 
                             1/2*b*np.cos(1/2*theta)*np.sin(nu) - np.cos(nu)*np.sin(nu)*np.sin(1/2*theta) ])
            ny = np.array([ (b*np.cos(nu)*np.cos(1/2*theta) + 2*(2*np.sin(nu)**2 - 1)*np.sin(1/2*theta))*np.cos(theta), 
                             (b*np.cos(nu)*np.cos(1/2*theta) + 2*(2*np.sin(nu)**2 - 1)*np.sin(1/2*theta))*np.sin(theta), 
                             b*np.cos(nu)*np.sin(1/2*theta) + 2*(2*np.cos(nu)**2 - 1)*np.cos(1/2*theta) ])
            normal = np.cross(nx,ny)
            X_normal.append(normal)
    Xf = np.matrix.flatten(Xdata)
    Yf = np.matrix.flatten(Ydata)
    Zf = np.matrix.flatten(Zdata)
    X = np.vstack((Xf,Yf,Zf)).T
    
    X_map = {}
    for i in range(N**2):
        v = X_normal[i]
        A = VectorToProjection(v)
        X_map[i] = A                
    X_check = np.zeros((N**2, 3+3*3))
    for i in range(N**2):
        X_check[i, 0:3] = X[i,:]
        X_check[i, 3:12] = np.reshape(gamma*X_map[i], (1,9))    

    X_check = gudhi.subsampling.sparsify_point_set(points = X_check, min_squared_dist = min_dist**2)
    X_check = np.array(X_check)
    N_points = X_check.shape[0]
    print('The sample contains '+repr(N_points)+ ' points.', flush=True)    
    
    return X_check, N_points

COLORS = ['red', 'green', 'blue', 'orange'] #colors of the persistence barcodes

def set_axes_equal(ax):
    '''
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input:
        ax (plt.axis): the axis.
    
    Remark:
        Copied from the anwser of karlo on https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    '''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
def PlotPointCloud(X, theta1 = 50, theta2 = 75, plot_axis = False, values = []):
    '''
    Plot a point cloud in 2D or 3D. The input point cloud X has to have dimension
    N x n with n = 2 or 3.
    
    Input:
        X (np.array): size N x n, a cloud of N points.
        theta1, theta2 (float, optionnal): camera angle (only in 3D).
        plot_axis (bool): whether to plot the axis (only in 3D).
        values (np.array, optional): is nonempty, color the plot according to values.
    '''
    n = np.shape(X)[1]     
    fig = plt.figure( figsize=(8,8) )
    if n==2:        
        if not plot_axis:
            plt.axis('off')
        if len(values) != 0:
            plot = plt.scatter(X[:,0],X[:,1], lw = 3, c=values)
            plt.colorbar(plot)
        else:
            plt.scatter(X[:,0],X[:,1], color = 'black', lw = 3)
        plt.axis('equal')
    elif n==3:
        ax = fig.add_subplot(111, projection='3d')
        if len(values) != 0:
            plot =         ax.scatter3D(X[:,0], X[:,1], X[:,2], lw = 3, c = values)

            plt.colorbar(plot)
        else:
            ax.scatter3D(X[:,0], X[:,1], X[:,2], color = 'black', lw = 3)
        if not plot_axis:
            plt.axis('off')
        set_axes_equal(ax)
        ax.view_init(theta1, theta2) 
        ax.xaxis.set_tick_params(labelsize=0)
        ax.yaxis.set_tick_params(labelsize=0)
        ax.zaxis.set_tick_params(labelsize=0)

def PlotPCA(X, theta1 = 50, theta2 = 75, plot_axis = True):
    '''
    Plot a point cloud in 3D after performing a Principal Component Analysis. 
    
    Input:
        X (np.array): size N x (n+n**2), a cloud of N points.
        theta1, theta2 (float, optionnal): camera angle (only in 3D).
        plot_axis (bool): whether to plot the axis.
    '''
    pca = PCA(n_components=3)
    pca.fit(X)
    X_pca = pca.transform(X)
    X_pca = np.transpose(X_pca)
    fig = plt.figure( figsize=(10,10) )
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(X_pca[0, :], X_pca[1, :], X_pca[2, :], color = 'black', lw = 3)
    if not plot_axis:
        plt.axis('off')
    set_axes_equal(ax)
    ax.view_init(theta1, theta2)
    ax.xaxis.set_tick_params(labelsize=0)
    ax.yaxis.set_tick_params(labelsize=0)
    ax.zaxis.set_tick_params(labelsize=0)

def PlotVectorField(X, theta1 = 50, theta2 = 75, plot_axis = True):
    '''
    Plot a vector field in 2D or 3D. The input point cloud X has to have dimension
    N x (n+n**2), and n is interpreted as the dimension of the space.
    The last n**2 coordinates are interpreted as projection matrix onto a line.
    
    Input:
        X (np.array): size N x (n+n**2), a cloud of N points.
        theta1, theta2 (float, optionnal): camera angle (only in 3D).
        plot_axis (bool): whether to plot the axis (only in 3D).
    '''
    N_points = np.shape(X)[0]
    n=int((-1+np.sqrt(1+4*np.shape(X)[1]))/2) #get the integer n such that n+n**2=np.shape(X)[1]

    if n==2:        
        l = .2   #arrows length
        fig = plt.figure( figsize=(10,10) )
        plt.scatter(X[:,0],X[:,1], color = 'black', lw = 3)
        for i in range(N_points):
            x = X[i,0:2]
            P = X[i,2:8]
            P = np.reshape(P, (2,2))
            v = ProjectionToVector(P)
            v = l*v
            plt.arrow(x[0], x[1], v[0], v[1], color = 'magenta',lw = 3)
            plt.arrow(x[0], x[1], -v[0], -v[1], color = 'magenta',lw = 3)
        plt.axis('off')
        plt.xlim((np.min(X[:,0])-l, np.max(X[:,0])+l))
        plt.ylim((np.min(X[:,1])-l, np.max(X[:,1])+l))

    if n==3:
        l = .5   #arrows length
        fig = plt.figure( figsize=(10,10) )
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(X[:,0], X[:,1], X[:,2], color = 'black', lw = 3)
        N = np.zeros((N_points,3))
        for i in range(N_points):
            P = X[i,3:13]
            P = np.reshape(P, (3,3))
            v = ProjectionToVector(P)
            N[i,:] = v
        ax.quiver(X[:,0], X[:,1], X[:,2], N[:,0], N[:,1], N[:,2], length=l, normalize=True, color = 'magenta')
        ax.quiver(X[:,0], X[:,1], X[:,2], -N[:,0], -N[:,1], -N[:,2], length=l, normalize=True, color = 'magenta')
        if not plot_axis:
            plt.axis('off')
        set_axes_equal(ax)
        ax.view_init(theta1, theta2)
        ax.xaxis.set_tick_params(labelsize=0)
        ax.yaxis.set_tick_params(labelsize=0)
        ax.zaxis.set_tick_params(labelsize=0)   
        

def PlotPersistenceDiagram(st, homology_coeff_field = 2):
    '''
    Displays the persistence diagram of the simplex tree st.
    
    Input:
        st (gudhi.SimplexTree).
        homology_coeff_field (int, optional): Field of coefficients to compute the persistence
    '''
    gudhi.plot_persistence_diagram(st.persistence(homology_coeff_field = homology_coeff_field), colormap = tuple(COLORS))            

        
def PlotPersistenceBarcodes(st, tmax = 1, d=2, eps = 0, hide_small_infinite_bars = False, 
                            barcode_index = [],  homology_coeff_field = 2, 
                            xtick = 0, xticks = []):
    '''
    Displays the persistence barcodes of the simplex tree st.
    
    Input:
        st (gudhi.SimplexTree).
        tmax (float, optional): maximal filtration value to plot the barcodes.
        d (int, optional): maximal dimension to compute the persistence.
        eps (float, optional): minimal length of displayed bars.
        hide_small_infinite_bars (bool, optional): if False, do not display bars of length < epsilon 
                                         which exceed the maximal filtration value.
        barcode_index (list of int, optional): array containing the dimensions to plot the barcodes. 
                                               By default: range(d+1).
        homology_coeff_field (int, optional): Field of coefficients to compute the persistence
        xticks (list of float, optional): a list of xticks. If nonempty, replace the default ticks by this list.
        xtick (bool, optional): if nonzero, add a xtick to the barcodes
    '''
    st.persistence(homology_coeff_field = homology_coeff_field)    #compute the persistence

    ' Plot the persistence barcodes '
    width = 0.75       #width of the bars
    eps1 = .1          #y margin
    eps2 = .0          #x margin
    if not barcode_index: 
        barcode_index = range(d+1)
    for i in barcode_index:
        diagram = st.persistence_intervals_in_dimension(i)
        if hide_small_infinite_bars:
            diagram = [t for t in diagram if min(t[1], tmax)-t[0] > eps]          #select large enough bars
            diagram = [[t[0], min(t[1], tmax)] for t in diagram]                  #threshold the bars exceeding tmax
        else:
            diagram = [t for t in diagram if t[1]-t[0] > eps]                     #select large enough bars
            diagram = [[t[0], min(t[1], tmax)] for t in diagram]                  #threshold the bars exceeding tmax
        if diagram:
            color = COLORS[i]
            plt.figure( figsize=(10,2) )
            ax = plt.axes(frameon=True)    
            for j in range(len(diagram)):
                t = diagram[j]
                plt.fill([t[0], t[1], t[1], t[0]], [j, j, j+width, j+width], fill=True, c=color, lw = 1.4)

            ax.set_xlim(-eps2, tmax+eps2)
            ax.spines['bottom'].set_visible(True)
            ax.spines['bottom'].set_bounds(0, tmax)
            ax.spines['bottom'].set_linewidth(1.5)  
            if xticks:
                plt.xticks(xticks)
            else:
                if xtick:  
                    plt.xticks([0, xtick, tmax])
                    ax.set_xticklabels([str(0), '%.1f' %xtick, '%.1f' %tmax])
                else:      
                    plt.xticks([0, tmax])
                    ax.set_xticklabels([str(0), '%.1f' %tmax])
            ax.tick_params(axis='both', which='major', labelsize=20, size = 9, width = 2) 
            ax.axes.get_yaxis().set_visible(False)
            ax.set_ylim(-eps1, len(diagram) + eps1)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)     
            plt.title('H'+str(i)+'-barcode')
            
def PlotLifebar(Lifebar, filtration_max, plot_lifebar_curve = False):
    '''
    Plot the lifebar corresponding to the array Lifebar. The lifebar is a bar that 
    is hatched until the value t_dagger, and then solid until the value filtration_max.
    The value t_dagger is defined as the first value for which I is nonzero.
    If plit_lifebar_curve == True, plot the actual computed lifebar, which can 
    takes values different from 0 or 1 because of the non-simpliciality of the map.
    
    Input:
        Lifebar (np.array): size 1xN.
        filtration_max (float): the maximal filtration value.
        plot_lifebar_curve (bool)
    '''
    # Parameters of the lifebar
    width = 0.5   #width bar
    eps1 = 0.1   #ymargin
    eps2 = 0.02   #xmargin

    I = np.linspace(0, filtration_max, len(Lifebar))
    t_dagger = I[(Lifebar!=0).argmax()]
    plt.figure( figsize=(10,1) )
    ax = plt.axes(frameon=True)    
    if t_dagger==0:    
        plt.fill([0, filtration_max, filtration_max, 0], [0, 0, width, width], fill=False, c=COLORS[1], lw = 1.4, hatch = '/')
    else:
        plt.fill([0, t_dagger, t_dagger, 0], [0, 0, width, width], fill=False, c=COLORS[1], lw = 1.4, hatch = '/')
        plt.fill([t_dagger, filtration_max, filtration_max, t_dagger], [0, 0, width, width], fill=True, c=COLORS[1], lw = 1.4)
    ax.set_xlim(-eps2, filtration_max+eps2)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_bounds(-eps2, filtration_max+eps2)
    ax.spines['bottom'].set_linewidth(2)  
    ax.tick_params(axis='both', which='major', labelsize=20, size = 9, width = 2) 
    ax.axes.get_yaxis().set_visible(False)
    ax.set_ylim(-eps1, width + eps1)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)     
    
    if plot_lifebar_curve:
        ax.plot(I,Lifebar*width, lw=5, c='black')  #plot of the actual computed lifebar, 
                                                   #which can takes values different from 0 or 1 
                                                   #because of the non-simpliciality of g
