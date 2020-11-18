'''
velour -> persistence
(list of functions)

Combinatorics of simplicial complexes:
    - GetPowerSet
    - CopySimplexTree
    - GetVerticesSimplexTree
    - GetSimplicesSimplexTree
    - GetNeighborsSimplexTree
    - IsFilteredSimplicialMap
    - BarycentricSubdivisionSimplex
    - BarycentricSubdivisionSimplexTree
    - MappingCylinderFiltration
    - MappingConeFiltration
    - MappingTorusFiltration
    - GetWeakSimplicialApproximation

Persistence:
    - GetBettiCurves
    - RipsComplex
    - AlphaComplex
    - WeightedRipsFiltrationValue
    - WeightedRipsFiltration
    - DTMFiltration
    - AlphaDTMFiltration

Bundle filtrations:
    - BundleFiltrationMaximalValue
    - TriangulateProjectiveSpace
    - GetFaceMapBundleFiltration
    - ComputeLifebar
'''

import gudhi
import numpy as np
import random
import itertools
from sklearn.metrics.pairwise import euclidean_distances

from .geometry import DTM, VectorToProjection, ProjectionToVector, IntersectionLineHyperplane

def GetPowerSet(L):
    '''
    Returns the powerset of the list L.
    
    Input:
        L (list).
        
    Output:
        L2 (list): the power set of L.
        
    Example:
        L = range(3)
        velour.GetPowerSet(L)
        [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]        
    '''
    L2 = list(itertools.chain.from_iterable(itertools.combinations(L, r) for r in range(len(L)+1)))
    return L2

def CopySimplexTree(st):
    '''
    Hard copy of a simplex tree.
    
    Input:
        st (gudhi.SimplexTree): the simplex tree to be copied.
    
    Output:
        st1 (gudhi.SimplexTree): a copy of the simplex tree.
        
    Example:
        st = gudhi.SimplexTree()
        print(st)
        ---> <gudhi.SimplexTree object at 0x7fded6968e30> 
        velour.CopySimplexTree(st)
        ---> <gudhi.SimplexTree at 0x7fded6968d90>
    '''
    st2 = gudhi.SimplexTree()
    for filtr in st.get_filtration():
        st2.insert(filtr[0], filtr[1])
    return st2

def GetVerticesSimplexTree(st):
    '''
    Returns the list of vertices of the simplex tree st.
    
    Input:
        st (gudhi.SimplexTree): the simplex tree.
        
    Output:
        Vertices (list of int): the vertices of st.
        
    Example:
        st = gudhi.SimplexTree()
        st.insert([0,2,258],0)
        velour.GetVerticesSimplexTree(st)
        ---> [0, 2, 258]
    '''
    Vertices = [simplex[0][0] for simplex in st.get_skeleton(0)]
    return Vertices

def GetSimplicesSimplexTree(st, dim):
    '''
    Returns the list of simplices of the simplex tree st of dimension dim (i.e.,
    of length dim+1).
    
    Input:
        st (gudhi.SimplexTree): the simplex tree.
        
    Output:
        Simplices (list of list of int): the simplices of st of dimension dim.
        
    Example:
        st = gudhi.SimplexTree()
        st.insert(range(3),0)
        velour.GetSimplicesSimplexTree(st, dim=1)
        ---> [[0, 1], [0, 2], [1, 2]]
    '''
    Simplices = [simplex[0] for simplex in st.get_skeleton(dim) if len(simplex[0])==dim+1]
    return Simplices

def GetNeighborsSimplexTree(st, v, t=np.inf, closed = True):
    '''
    Returns the list of neighbors of the vertex v in the simplex tree st, 
    at time t. If closed == True, v itself is considered as a neighbor.
    
    Input:
        st (gudhi.SimplexTree): the simplex tree.
        v (int): the vertex.
        t (float, optional): the time at which we consider the simplicial complex.
        closed (bool, optional): whether to count v itself in the neighbors.
        
    Output:
        Neighbors (list of int): the neighbors of v.
    
    Example:
        st = gudhi.SimplexTree()
        st.insert([0,1],0)
        st.insert([1,2],0)
        v = 0
        velour.GetNeighborsSimplexTree(st, v)
        ---> [1, 0]
    '''
    Neighbors = []
    Edges = st.get_cofaces([v],1) #get the edges containing v
    for filtr in Edges:
        if filtr[1] <= t:
            simplex = filtr[0]
            simplex.remove(v)
            Neighbors.append(simplex[0])    
    if closed:
        Neighbors.append(v)
    return Neighbors

def IsFilteredSimplicialMap(st1,st2,g):
    '''
    Tests whether a map g: st1 --> st2 between vertex sets of two simplex trees is a simplicial map.
    
    Input:
        st1 (gudhi.SimplexTree): simplex tree, domain of g.
        st2 (gudhi.SimplexTree): simplex tree, codomain of g.
        g (dict int:int): a map between vertex sets of st1 and st2.
    '''
    time_not_simplicial = -1
    for filtr in st1.get_filtration():
        face = filtr[0]
        gface = [g[v] for v in face]
        gface = list(np.unique(gface))
        t = st2.filtration(gface)
        if t == np.inf:
            time_not_simplicial = filtr[1]
            break
    if time_not_simplicial == -1:
        print('The map is simplicial all along the filtration.', flush=True)
    else:
        print('The map is not simplicial from t = '+repr(time_not_simplicial)+'.', flush=True)

def BarycentricSubdivisionSimplex(Vertices):
    '''
    Subdivises barycentrically the simplex L. The resulting simplicial complex 
    has 2^n-1 vertices and n! maximal faces. 
    The vertices of the subdivision are the elements of the power set of the 
    input vertices (geometrically, such a vertex represents the barycenter of the vertices).
    The maximal faces of the subdivision are given as permutations of the input 
    vertices. The permutation 021 corresponds to the simplex [(0), (0,2), (0,2,1)].
    
    Input: 
        Vertices (list of int): length n, representing a (n-1)-simplex.
    
    Output: 
        MaximalFaces (list of int): length n!, representing the maximal faces of 
                                    the barycentric subdivision of the simplex
        NewVertices (list of int): length 2^n-1. If v is a vertex of the subdivision, 
                                    NewVertices[v] is the list of input vertices 
                                    that represents v.
                                    
    Example:
        Vertices = range(3)   #vertices of a triangle
        MaximalFaces, NewVertices = velour.BarycentricSubdivisionSimplex(Vertices)
        MaximalFaces
        ---> [[0, 3, 6], [0, 4, 6], [1, 3, 6], [1, 5, 6], [2, 4, 6], [2, 5, 6]]
        NewVertices
        ---> [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
    '''
    NewVertices = GetPowerSet(Vertices)
    del NewVertices[0]
    # the vertices of the subdivision are the elements of the power set of the input vertices
    DictNewVertices = {tuple(NewVertices[i]):i for i in range(len(NewVertices))}
    # a dict that indexes the new vertices. (0,): 0, (1,): 1, (0, 1): 2, ...
    Permutations = list(itertools.permutations(Vertices))
    # the maximal faces of the subdivision are given as permutations of the input vertices.
    MaximalFaces = []
    for perm in Permutations:
        face = [tuple(sorted(perm[:i+1])) for i in range(len(perm))]
        facedict=[DictNewVertices[v] for v in face]
        MaximalFaces.append(facedict)
    # we compute the index (in NewVertices) of the vertices of each maximal face
    # elements of MaximalFaces are [0, 2, 6], [0, 4, 6], ...    
    return MaximalFaces, NewVertices

def BarycentricSubdivisionSimplexTree(st, X=[], bool_dict=False):
    '''
    Subdivises barycentrically the simplex tree st. It consists in subdivising 
    each of its simplices.
    If bool_dict == True, returns a dict that gives the power-set form of the 
    vertices (see function BarycentricSubdivisionSimplex). 

    Input: 
        st (gudhi.simplex_tree): the simplex tree to subdivise.
        X (np.array, optional): size NxM, representing the coordinates of the 
                                vertices of the simplicial complex.
        bool_dict (bool, optional): whether to return the dict.
    
    Output: 
        st_sub (gudhi.simplex_tree): the subdivised simplex tree.
        dictNewVertices (dict tuples:int)
        Y (optional, if X!=[]): a N'xM np.array, representing the vertices of 
                                the new simplicial complex, according to the shadow map.
                                
    Example:
        st = gudhi.SimplexTree()
        st.insert([0,1,2], 0)
        st.remove_maximal_simplex([0,1,2])   #st is a circle
        st_sub, dictNewVertices = velour.BarycentricSubdivisionSimplexTree(st, bool_dict = True)
        st_sub.num_vertices()
        ---> 6
        dictNewVertices
        ---> {(0,): 0, (1,): 1, (0, 1): 2, (2,): 3, (0, 2): 4, (1, 2): 5}

    Example:
        st = gudhi.SimplexTree()
        st.insert([0,1], 0)
        X = np.array([[1,0],[0,1]])
        st_sub, Y = velour.BarycentricSubdivisionSimplexTree(st, X=X)
        ---> The new simplex tree has 3 vertices (previously 2).
        Y
        ---> array([[1. , 0. ],
                    [0. , 1. ],
                    [0.5, 0.5]])
    '''
    st_sub = gudhi.SimplexTree()
    dictNewVertices = {}
    l=0
    
    for filtr in st.get_filtration():
        simplex = filtr[0]
        time = filtr[1]
        MaximalFaces, NewVertices = BarycentricSubdivisionSimplex(simplex)
        for v in NewVertices:
            if tuple(v) not in dictNewVertices:
                dictNewVertices[tuple(v)] = l 
                l+=1
        for face in MaximalFaces:
            faceNew = [NewVertices[w] for w in face]
            faceDict = [dictNewVertices[tuple(w)] for w in faceNew]
            st_sub.insert(faceDict, time)

    if bool_dict:
        return st_sub, dictNewVertices
    elif np.size(X)==0:
        return st_sub
    else:
        dictNewVerticesInv = {dictNewVertices[v]:v for v in dictNewVertices}
        Y = np.zeros((len(dictNewVerticesInv), np.shape(X)[1]))
        for v in dictNewVerticesInv:
            Y[v] = np.mean([X[i,:] for i in dictNewVerticesInv[v]],0)
        result_str = 'Subdivised Complex is of dimension ' + repr(st_sub.dimension()) + ' (previously '+ repr(st.dimension()) + ') - ' + \
                        repr(st_sub.num_simplices()) + ' simplices (previously '+ repr(st.num_simplices())+ ') - ' + \
                        repr(st_sub.num_vertices()) + ' vertices (previously '+ repr(st.num_vertices())+ ').'
        print(result_str,  flush=True)        
        return st_sub, Y

def MappingCylinderFiltration(st_X, st_Y, g):
    '''
    Creates a filtration of the mapping cylinder of g: st_X --> st_Y.
        
    Input:
        st_X (gudhi.SimplexTree): simplex tree, domain of g.
        st_Y (gudhi.SimplexTree): simplex tree, codomain of g.
        g (dict int:int): a simplicial map { (vertices of st_X):(vertices of st_Y)}.
    
    Output: 
        st_cyl (gudhi.SimplexTree): the mapping cylinder of g. 
    
    Example:
        st_X = gudhi.SimplexTree()
        st_X.insert([0,1,2], 0)
        st_X.remove_maximal_simplex([0,1,2])   #st is sphere
        g = {0:0,1:1,2:2}
        st_cyl = velour.MappingCylinderFiltration(st_X, st_X, g)   #st_cyl is a cylinder
        st_cyl.persistence(persistence_dim_max=True, homology_coeff_field = 2)
        ---> [(1, (0.0, inf)), (0, (0.0, inf))] 
    '''
    NewVerticesList = [tuple([v,0]) for v in GetVerticesSimplexTree(st_X)]+[tuple([v,1]) for v in GetVerticesSimplexTree(st_Y)]
    NewVertices = {NewVerticesList[i]:i for i in range(len(NewVerticesList))}

    st_cyl = gudhi.SimplexTree()

    #insert st_X
    for filtr in st_X.get_filtration():
        simplex = filtr[0]
        simplex_cyl = [NewVertices[tuple([v,0])] for v in simplex]
        st_cyl.insert(simplex_cyl, filtr[1])
    
    #insert st_Y
    for filtr in st_Y.get_filtration():
        simplex = filtr[0]
        simplex_cyl = [NewVertices[tuple([v,1])] for v in simplex]
        st_cyl.insert(simplex_cyl, filtr[1])
    
    #connect st_X to st_Y
    for filtr in st_X.get_filtration():
        simplex_X = filtr[0]
        simplex_Y = [g[v] for v in simplex_X]
        simplex_X_cyl = [NewVertices[tuple([v,0])] for v in simplex_X]
        simplex_Y_cyl = [NewVertices[tuple([v,1])] for v in simplex_Y]
        simplex_cyl = simplex_X_cyl+simplex_Y_cyl   #union of the simplices
        simplex_cyl = list(set(simplex_cyl))        #unique
        st_cyl.insert(simplex_cyl, filtr[1])
        
    result_str = 'Mapping Cylinder Complex is of dimension ' + repr(st_cyl.dimension()) + ' - ' + \
                 repr(st_cyl.num_simplices()) + ' simplices - ' + \
                 repr(st_cyl.num_vertices()) + ' vertices.'
    print(result_str, flush=True)
        
    return st_cyl

def MappingConeFiltration(st_X, st_Y, g, filtration_max=0.5):
    '''
    Creates a filtration of the mapping cone of g: st_X --> st_Y.
   
    Input:
        st_X (gudhi.SimplexTree): simplex tree, domain of g.
        st_Y (gudhi.SimplexTree): simplex tree, codomain of g.
        g (dict int:int): a simplicial map { (vertices of st_X):(vertices of st_Y)}.
        filtration_max (float): maximal filtration value.
    
    Output: 
        st_cone (gudhi.SimplexTree): the mapping cone of g. 
        
    Example:
        st_X = gudhi.SimplexTree()
        st_X.insert([0,1,2], 0)
        st_X.remove_maximal_simplex([0,1,2])   #st_X is a circle
        st_Y = gudhi.SimplexTree()
        st_Y.insert([0,1,2], 0)   #st_Y is a disk
        g = {0:0,1:1,2:2}
        st_cone = velour.MappingConeFiltration(st_X, st_Y, g)    #st_cone is a sphere
        st_cone.persistence(persistence_dim_max=True, homology_coeff_field = 2)
        ---> [(2, (0.0, inf)), (0, (0.0, inf))]
    '''   
    st_X = CopySimplexTree(st_X)
    st_X.prune_above_filtration(filtration_max)

    NewVerticesList = [tuple([v,0]) for v in GetVerticesSimplexTree(st_X)]+[tuple([v,1]) for v in GetVerticesSimplexTree(st_Y)]
    NewVerticesList += [-1] #add coning point
    NewVertices = {NewVerticesList[i]:i for i in range(len(NewVerticesList))}

    st_cone = gudhi.SimplexTree()

    #insert st_X
    for filtr in st_X.get_filtration():
        simplex = filtr[0]
        simplex_cyl = [NewVertices[tuple([v,0])] for v in simplex]
        st_cone.insert(simplex_cyl, filtr[1])
    
    #insert st_Y
    for filtr in st_Y.get_filtration():
        simplex = filtr[0]
        simplex_cyl = [NewVertices[tuple([v,1])] for v in simplex]
        st_cone.insert(simplex_cyl, filtr[1])
    
    #connect st_X to st_Y according to g
    for filtr in st_X.get_filtration():
        simplex_X = filtr[0]
        simplex_Y = [g[v] for v in simplex_X]
        simplex_X_cyl = [NewVertices[tuple([v,0])] for v in simplex_X]
        simplex_Y_cyl = [NewVertices[tuple([v,1])] for v in simplex_Y]
        simplex_cyl = simplex_X_cyl+simplex_Y_cyl   #union of the simplices
        simplex_cyl = list(set(simplex_cyl))        #unique
        st_cone.insert(simplex_cyl, filtr[1])
    
    #coning
    for filtr in st_X.get_filtration():
        simplex = filtr[0]
        simplex_cyl = [NewVertices[tuple([v,0])] for v in simplex]
        simplex_cone = simplex_cyl+[NewVertices[-1]]
        st_cone.insert(simplex_cone, filtr[1])
    
    result_str = 'Mapping Cone Complex is of dimension ' + repr(st_cone.dimension()) + ' - ' + \
                 repr(st_cone.num_simplices()) + ' simplices - ' + \
                 repr(st_cone.num_vertices()) + ' vertices.'
    print(result_str, flush=True)
    
    return st_cone

def MappingTorusFiltration(st_X, g):
    '''
     Creates a filtration of the mapping cone of g: st_X --> st_Y.
   
    Input:
        st_X (gudhi.SimplexTree): simplex tree.
        g (dict int:int): a simplicial map { (vertices of st_X):(vertices of st_X)}.
    
    Output: 
        st_tor (gudhi.SimplexTree): the mapping torus of g. 

    Example:
        st_X = gudhi.SimplexTree()
        st_X.insert([0,1,2], 0)
        st_X.remove_maximal_simplex([0,1,2])   #st_X is a circle
        g = {0:0,1:1,2:2}
        st_tor = velour.MappingTorusFiltration(st_X, g)   #st_tor is a torus
        st_tor.persistence(persistence_dim_max=True, homology_coeff_field = 3)
        ---> [(2, (0.0, inf)), (1, (0.0, inf)), (1, (0.0, inf)), (0, (0.0, inf))]

    Example:
        st_X = gudhi.SimplexTree()
        st_X.insert([0,1,2], 0)
        st_X.remove_maximal_simplex([0,1,2])   #st_X is a circle
        g = {0:0,1:2,2:1}
        st_tor = velour.MappingTorusFiltration(st_X, g)   #st_tor is a Klein bottle
        st_tor.persistence(persistence_dim_max=True, homology_coeff_field = 3)
        ---> [(1, (0.0, inf)), (0, (0.0, inf))]
    '''
    NewVerticesList = [tuple([v,0]) for v in GetVerticesSimplexTree(st_X)]+[tuple([v,1]) for v in GetVerticesSimplexTree(st_X)]+[tuple([v,2]) for v in GetVerticesSimplexTree(st_X)]
    NewVertices = {NewVerticesList[i]:i for i in range(len(NewVerticesList))}
    st_tor = gudhi.SimplexTree()

    #insert st_X at 0, 1 and 2
    for filtr in st_X.get_filtration():
        simplex = filtr[0]
        simplex_cyl = [NewVertices[tuple([v,0])] for v in simplex]
        st_tor.insert(simplex_cyl, filtr[1])
        simplex_cyl = [NewVertices[tuple([v,1])] for v in simplex]
        st_tor.insert(simplex_cyl, filtr[1])
        simplex_cyl = [NewVertices[tuple([v,2])] for v in simplex]
        st_tor.insert(simplex_cyl, filtr[1])
    
    #connect st_X at 0 to st_X at 1 and 
    #connect st_X at 1 to st_X at 2 according to identity
    for filtr in st_X.get_filtration():
        simplex_X = filtr[0]
        simplex_X0 = [NewVertices[tuple([v,0])] for v in simplex_X]
        simplex_X1 = [NewVertices[tuple([v,1])] for v in simplex_X]
        simplex_X2 = [NewVertices[tuple([v,2])] for v in simplex_X]
        simplex_X01 = simplex_X0+simplex_X1
        simplex_X12 = simplex_X1+simplex_X2
        st_tor.insert(simplex_X01, filtr[1])
        st_tor.insert(simplex_X12, filtr[1])

            
    #connect st_X at 2 to st_X at 0 according to g
    for filtr in st_X.get_filtration():
        simplex_X = filtr[0]
        simplex_X2 = [NewVertices[tuple([v,2])] for v in simplex_X]
        simplex_fX = [g[v] for v in simplex_X]
        simplex_fX0 = [NewVertices[tuple([v,0])] for v in simplex_fX]
        simplex_tor = simplex_X2+simplex_fX0   #union of the simplices
        simplex_tor = list(set(simplex_tor))   #unique
        st_tor.insert(simplex_tor, filtr[1])
    
    result_str = 'Mapping Torus Complex is of dimension ' + repr(st_tor.dimension()) + ' - ' + \
                 repr(st_tor.num_simplices()) + ' simplices - ' + \
                 repr(st_tor.num_vertices()) + ' vertices.'
    print(result_str,  flush=True)
 
    return st_tor

def GetWeakSimplicialApproximation(st_X, st_Y, FaceMap, filtration_max=0.1):
    '''
    Returns a random weak approximation to the map (FaceMap: st_X^{(0)} --> st_Y). 
    If FaceMap does not satisfies the weak star condition, returns False. 
    It first computes the dictionaries CorrespondingSimplices and AdmissibleVertices.
    CorrespondingSimplices is a dictionary { (vertex of st_X):(list of simplices of st_Y) }, 
    where each simplex is the image of the neighbors of the vertex of st by the minimal face map.
    AdmissibleVertices is a dictionary { (vertex of st_X):(list of vertices of st_Y) }, 
    where the list contains the vertices of st_Y which satisfies the weak star condition for the vertex in st_X.
    The weak approximation is selected by picking for each vertex of st_X a random admissible vertex of st_Y.
    
    Input: 
        st_X (gudhi.SimplexTree): simplex tree, domain of FaceMap.
        st_Y (gudhi.SimplexTree): simplex tree, codomain of FaceMap.
        FaceMap (dic int:(list of int): a map (vertex of st_X):(simplex of st_Y).
        filtration_max (float): maximal time value to compute the weak simplicial approximation.

    Output:
        RandomChoiceAdmissibleVertices (dict int:int): a map (vertex of st_X):(vertex of st_Y),
                                                       weak simplicial approximation to FaceMap.
    Example:
        st_X = gudhi.SimplexTree()
        st_X.insert([0], 0)
        st_Y = gudhi.SimplexTree()
        st_Y.insert([0,1], 0)
        FaceMap = {0:[0,1]}
        g = velour.GetWeakSimplicialApproximation(st_X, st_Y, FaceMap, filtration_max=0.01)
        ---> The map satisfies the weak star condition.
        g 
        ---> {0: 0}
        
    Example:
        st_X = gudhi.SimplexTree()
        st_X.insert([0,1,2], 0)   #st_Y is a triangle
        st_Y = gudhi.SimplexTree()
        st_Y.insert([0,1,2], 0)
        st_Y.remove_maximal_simplex([0,1,2])   #st_Y is a circle
        FaceMap = {0:[0], 1:[1], 2:[2]}
        velour.GetWeakSimplicialApproximation(st_X, st_Y, FaceMap, filtration_max=0.01)
        ---> Impossible choice: the vertex 2 does not satisfies the weak star condition.
    '''   
    st_X = CopySimplexTree(st_X)
    st_X.prune_above_filtration(filtration_max)
    
    #Get CorrespondingSimplices
    CorrespondingSimplices = {}
    for v in GetVerticesSimplexTree(st_X):                #Compute the neighbors
        Neighbors = GetNeighborsSimplexTree(st_X, v)
        Simplices = []
        for w in Neighbors:                            #Compute the corresponding simplices
            Simplices.append(FaceMap[w])     
        CorrespondingSimplices[v] = Simplices                             
    
    #Get AdmissibleVertices
    AdmissibleVertices = {}   
    for v in GetVerticesSimplexTree(st_X):  
        AdmissibleVerticesv = []
        for w in GetVerticesSimplexTree(st_Y):  
            is_w_included = True
            for simplex in CorrespondingSimplices[v]:
                if not w in set(simplex):
                    is_w_included = False
            if is_w_included:
                AdmissibleVerticesv.append(w)
        AdmissibleVertices[v] = AdmissibleVerticesv

    #Star condition, i.e. tests whether each vertex of st_X admits a vertex in st_Y
    test_star_condition = True
    for v in GetVerticesSimplexTree(st_X):  
        if not AdmissibleVertices[v]:            
            test_star_condition = False
            vertex_issue = v             #a vertex which does not satisfies the star condition

    if test_star_condition:   
        print('The map satisfies the weak star condition.', flush=True)
        #Get a random weak approximation
        RandomChoiceAdmissibleVertices = {}    
        for v in GetVerticesSimplexTree(st_X):
            RandomChoiceAdmissibleVertices[v] = random.choice(AdmissibleVertices[v])
        return RandomChoiceAdmissibleVertices
    else:
        print('Impossible choice: the vertex '+repr(vertex_issue)+' does not satisfies the weak star condition.', flush = True)
        return False

def GetBettiCurves(st, I, homology_coeff_field = 2, dim=0):
    '''
    Computes the Betti curves of the simplex tree st, on the interval 
    I, up to dimension dim.
    If dim=0, compute the Betti curves up to the dimension of st.

    Input:
        st (gudhi.SimplexTree): simplex tree whose Betti curves are to be computed
        I (np.array): interval. Shape 1xN. 
        homology_coeff_field (int, optional): field for computing persistent homology
        dim (int, optional): maximal dimension to compute the Betti curves. 
                             If dim=0, the dimension of st is chosen.
    
    Output:
        BettiCurves (np.array): the Betti curves. Shape (dim)x100. The ith 
                                Betti curve is given by BettiCurve[i,:].  
                                
    Remark:
        Optimization by Marc Glisse in https://github.com/GUDHI/gudhi-devel/pull/423/files

    Example:
        st = gudhi.SimplexTree()
        st.insert(range(4),0)
        st.remove_maximal_simplex(range(4)) #st is a sphere
        I = np.array([0, 0.5, 1])
        velour.GetBettiCurves(st, I)
        ---> array([[1., 1., 1.],
                    [0., 0., 0.],
                    [1., 1., 1.]])
    '''
    if dim == 0:
        dim = st.dimension()
        
    st.persistence(persistence_dim_max=True, homology_coeff_field = homology_coeff_field)
    Diagrams = [st.persistence_intervals_in_dimension(i) for i in range(dim+1)]

    BettiCurves = []
    step_x = I[1]-I[0]

    for diagram in Diagrams:
        bc =  np.zeros(len(I))
        if diagram.size != 0:
            diagram_int = np.clip(np.ceil((diagram[:,:2] - I[0]) / step_x), 0, len(I)).astype(int)
            for interval in diagram_int:
                bc[interval[0]:interval[1]] += 1
        BettiCurves.append(np.reshape(bc,[1,-1]))
        
    return np.reshape(BettiCurves, (dim+1, len(I)))

def RipsComplex(X, filtration_max=np.inf, dimension_max=3):
    '''
    Build the Rips filtration over X, until time filtration_max.
        
    Input: 
        X (np.array): size (N x M), representing a point cloud of R^M.
        filtration_max (float): filtration maximal value.
        dimension_max (int): maximal dimension to expand the complex. 
                             Must be k+1 to read k homology
    
    Output: 
        st (gudhi.SimplexTree): the Rips filtration of X.
        
    '''   
    N = X.shape[0]
    distances = euclidean_distances(X)                       #pairwise distances
    st = gudhi.SimplexTree()                                 #create an empty simplex tree        
    for i in range(N):                                       #add vertices to the simplex tree
            st.insert([i], filtration = 0)         
    distances_threshold = distances<2*filtration_max
    indices = zip(*distances_threshold.nonzero())
    for u in indices:                                        #add edges to the simplex tree
        i = u[0]; j = u[1]
        if i<j:                                        #add only edges [i,j] with i<j
            st.insert([i,j], filtration  = distances[i,j]/2)
    st.expansion(dimension_max)                              #expand the flag complex 
    result_str = 'Rips Complex is of dimension ' + repr(st.dimension()) + ' - ' + \
        repr(st.num_simplices()) + ' simplices - ' + \
        repr(st.num_vertices()) + ' vertices.' +\
        ' Filtration maximal value is ' + repr(filtration_max) + '.'
    print(result_str, flush=True)      
    return st

def AlphaComplex(X, filtration_max = np.inf):
    '''
    Build the Delaunay filtration over X, until time filtration_max.
        
    Input: 
        X (np.array): size (N x M), representing a point cloud of R^M.
        filtration_max (float): filtration maximal value.
    
    Output: 
        st_alpha (gudhi.SimplexTree): the Delaunay filtration of X.
        
    Example:
        X = SampleOnCircleNormalCheck(N = 30, gamma=1, sd=0)
        st = velour.AlphaComplex(X)
        ---> Alpha-complex is of dimension 5 - 8551 simplices - 30 vertices.
    '''   
    st_alpha = gudhi.AlphaComplex(X).create_simplex_tree()   #create an alpha-complex
    st_alpha.prune_above_filtration(filtration_max)
    result_str = 'Alpha-complex is of dimension ' + repr(st_alpha.dimension()) + ' - ' + \
        repr(st_alpha.num_simplices()) + ' simplices - ' + \
        repr(st_alpha.num_vertices()) + ' vertices.'
    print(result_str, flush=True)  
    return st_alpha

def WeightedRipsFiltrationValue(p, fx, fy, d, n = 10):
    '''
    Computes the filtration value of the edge [x,y] in the weighted Rips filtration.
    If p is not 1, 2 or 'np.inf, an implicit equation is solved.
    The equation to solve is G(I) = d, where G(I) = (I**p-fx**p)**(1/p)+(I**p-fy**p)**(1/p).
    We use a dichotomic method.
    
    Input:
        p (float): parameter of the weighted Rips filtration, in [1, +inf) or np.inf
        fx (float): filtration value of the point x
        fy (float): filtration value of the point y
        d (float): distance between the points x and y
        n (int, optional): number of iterations of the dichotomic method
        
    Output: 
        val (float): filtration value of the edge [x,y], i.e. solution of G(I) = d.
    
    Example:
        WeightedRipsFiltrationValue(2.4, 2, 3, 5, 10)
    '''
    if p==np.inf:
        value = max([fx,fy,d/2])
    else:
        fmax = max([fx,fy])
        if d < (abs(fx**p-fy**p))**(1/p):
            value = fmax
        elif p==1:
            value = (fx+fy+d)/2
        elif p==2:
            value = np.sqrt( ( (fx+fy)**2 +d**2 )*( (fx-fy)**2 +d**2 ) )/(2*d)            
        else:
            Imin = fmax; Imax = (d**p+fmax**p)**(1/p)
            for i in range(n):
                I = (Imin+Imax)/2
                g = (I**p-fx**p)**(1/p)+(I**p-fy**p)**(1/p)
                if g<d:
                    Imin=I
                else:
                    Imax=I
            value = I
    return value

def WeightedRipsFiltration(X, F, p, dimension_max =2, filtration_max = np.inf):
    '''
    Compute the weighted Rips filtration of a point cloud, weighted with the 
    values F, and with parameter p. Requires sklearn.metrics.pairwise.euclidean_distances
    to compute pairwise distances between points.
    
    Input:
        X (np.array): size Nxn, representing N points in R^n.
        F (np.array):size 1xN,  representing the values of a function on X.
        p (float): a parameter in [0, +inf) or np.inf.
        dimension_max (int, optional): maximal dimension to expand the complex.
        filtration_max (float, optional): maximal filtration value of the filtration.
    
    Output:
        st (gudhi.SimplexTree): the weighted Rips filtration. 
    '''
    N_tot = X.shape[0]     
    distances = euclidean_distances(X)          #compute the pairwise distances
    st = gudhi.SimplexTree()                    #create an empty simplex tree
    for i in range(N_tot):                      #add vertices to the simplex tree
        value = F[i]
        if value<filtration_max:
            st.insert([i], filtration = F[i])            
    for i in range(N_tot):                      #add edges to the simplex tree
        for j in range(i):
            value = WeightedRipsFiltrationValue(p, F[i], F[j], distances[i][j])
            if value<filtration_max:
                st.insert([i,j], filtration  = value)    
    st.expansion(dimension_max)                 # expand the simplex tree 
    result_str = 'Weighted Rips Complex is of dimension ' + repr(st.dimension()) + ' - ' + \
        repr(st.num_simplices()) + ' simplices - ' + \
        repr(st.num_vertices()) + ' vertices.' +\
        ' Filtration maximal value is ' + str(filtration_max) + '.'
    print(result_str, flush=True)
    return st

def DTMFiltration(X, m, p, dimension_max =2, filtration_max = np.inf):
    '''
    Computes the DTM-filtration of a point cloud, with parameters m and p.
    
    Input:
        X (np.array): size Nxn, representing N points in R^n.
        m (float): parameter of the DTM, in [0,1). 
        p (float): parameter of the DTM-filtration, in [0, +inf) or np.inf.
        dimension_max (int, optional): maximal dimension to expand the complex.
        filtration_max (float, optional): maximal filtration value of the filtration.
    
    Output:
        st (gudhi.SimplexTree): the DTM-filtration. 
    '''    
    DTM_values = DTM(X,X,m)
    st = WeightedRipsFiltration(X, DTM_values, p, dimension_max, filtration_max)
    return st

def AlphaDTMFiltration(X, m, p, dimension_max =2, filtration_max = np.inf):
    '''
    /!\ this is a heuristic method, that speeds-up the computation.
    It computes the DTM-filtration seen as a subset of the Delaunay filtration.
    
    Input:
        X (np.array): size Nxn, representing N points in R^n.
        m (float): parameter of the DTM, in [0,1). 
        p (float): parameter of the DTM-filtration, in [0, +inf) or np.inf.
        dimension_max (int, optional): maximal dimension to expand the complex.
        filtration_max (float, optional): maximal filtration value of the filtration.
    
    Output:
        st (gudhi.SimplexTree): the alpha-DTM filtration.
    '''
    N_tot = X.shape[0]     
    alpha_complex = gudhi.AlphaComplex(points=X)
    st_alpha = alpha_complex.create_simplex_tree()    
    Y = np.array([alpha_complex.get_point(i) for i in range(N_tot)])
    distances = euclidean_distances(Y)             #computes the pairwise distances
    DTM_values = DTM(X,Y,m)                        #/!\ in 3D, gudhi.AlphaComplex may change the ordering of the points
    
    st = gudhi.SimplexTree()                       #creates an empty simplex tree
    for simplex in st_alpha.get_skeleton(2):       #adds vertices with corresponding filtration value
        if len(simplex[0])==1:
            i = simplex[0][0]
            st.insert([i], filtration  = DTM_values[i])
        if len(simplex[0])==2:                     #adds edges with corresponding filtration value
            i = simplex[0][0]
            j = simplex[0][1]
            value = WeightedRipsFiltrationValue(p, DTM_values[i], DTM_values[j], distances[i][j])
            st.insert([i,j], filtration  = value)
    st.expansion(dimension_max)                    #expands the complex
    result_str = 'Alpha Weighted Rips Complex is of dimension ' + repr(st.dimension()) + ' - ' + \
        repr(st.num_simplices()) + ' simplices - ' + \
        repr(st.num_vertices()) + ' vertices.' +\
        ' Filtration maximal value is ' + str(filtration_max) + '.'
    print(result_str)
    return st

def BundleFiltrationMaximalValue(X, filtration = 'Rips', n=0, m=0): 
    '''
    Returns the filtration maximal value of the Vietoris-Rips or Cech bundle 
    filtration of X. It is equal to D/sqrt(2) (Rips) or D (Cech), where D is the 
    smallest distance between points of X and the medial axis of the Grassmann 
    manifold G_1(R^m) seen in M(R^m).
    If n=0, computes automatically n such that np.shape(X)[1] = n+n^2
        
    Input: 
        X (np.array): size (N x (n+m^2)), representing a point cloud.  
        filtration (str, optional): the type of filtration.
        n,m (int, optional): the dimensions of the spaces.
    
    Output: 
        max_value (float): the filtration maximal value.
        
    Example:
        X = velour.SampleOnCircleNormalCheck(N = 30, gamma=1, sd=0)
        filtration_max = velour.BundleFiltrationMaximalValue(X, filtration='Cech')
        ---> Filtration maximal value is 0.707.
        filtration_max = velour.BundleFiltrationMaximalValue(X, filtration='Rips')
        ---> Filtration maximal value is 0.500.
    '''   
    N = np.shape(X)[0]    
    if n==0:
        n=int((-1+np.sqrt(1+4*np.shape(X)[1]))/2) #get the integer n such that n+n**2=np.shape(X)[1]
        m=n
    DistancesToMed = np.zeros((N,1))    
    for i in range(N):
        A = np.reshape(X[i, n:(n+m**2)], (m,m))
        S = (A+A.transpose())/2
        Eigenvalues, Eigenvectors = np.linalg.eig(S)
        Eigenvalues_sorted = Eigenvalues[np.argsort(Eigenvalues)[::-1]]
        DistancesToMed[i] = np.abs(Eigenvalues_sorted[0]-Eigenvalues_sorted[1])/np.sqrt(2)
        # Distance between A and the medial axis
    if filtration=='Cech':
        max_value = min(DistancesToMed)
    elif filtration=='Rips':
        max_value = min(DistancesToMed)/np.sqrt(2)
    max_value = max_value[0]
    print('Filtration maximal value is '+repr(max_value)+'.', flush=True)
    return max_value

def TriangulateProjectiveSpace(d = 2):
    '''
    Gives a triangulation st_grass of the projective space of dimension d, 
    map_grass the coordinates of its vertices (seen as dxd matrices), triangulation 
    st_sphere of the sphere of dimension d, map_sphere the coordinates of its 
    vertices (seen a dxd matrices), and a dictionary RepresentativeSet which is 
    a choice of representative vertices of Sphere for the relation Sphere/~ = ProjectiveSpace
        
    Input: 
        d (int): the dimension of the projective space.
    
    Output: 
        st_grass (gudhi.SimplexTree): triangulation of the projective space.
        map_grass (dict (vertex of st_grass}: (dxd np.array)): coordinates of the vertices of st_grass.
        st_sphere (gudhi.SimplexTree): triangulation of the sphere.
        Sphere_map (dict (vertex of st_sphere}:(dxd np.array)): the coordinates of the vertices of st_sphere.
        RepresentativeSet (dict (vertex of st_sphere):(vertex of st_sphere)): a choice of representatives for the relation Sphere/~ = ProjectiveSpace.
    
    Example:
        st_grass, map_grass, st_sphere, map_sphere, RepresentativeSet = velour.TriangulateProjectiveSpace(3)
        st_grass.persistence(persistence_dim_max=True, homology_coeff_field = 2)
        ---> [(3, (0.0, inf)), (2, (0.0, inf)), (1, (0.0, inf)), (0, (0.0, inf))]
        st_grass.persistence(persistence_dim_max=True, homology_coeff_field = 3)
        ---> [(3, (0.0, inf)), (0, (0.0, inf))]

    '''
    # Build the simplicial complex
    st_sphere = gudhi.SimplexTree()
    st_sphere.insert(range(d+2),0)                           #simplex of dimension d+1    
    st_sphere.remove_maximal_simplex(range(d+2))             #triangulation of the sphere of dimension d
    st_sphere, dictNewVertices = BarycentricSubdivisionSimplexTree(st_sphere, bool_dict = True) #barycentric subdivision
    dictNewVerticesInv = {dictNewVertices[v]:v for v in dictNewVertices}
    
    VerticesSimplex = {i for i in range(d+2)}          #set of vertices before subdivision

    EquivalenceRelation = {}                           #dict representing the equivalence relation Sphere/~ = ProjectiveSpace
    for t in dictNewVertices:
        EquivalenceRelation[t] = tuple(VerticesSimplex.difference(t))
        
    RepresentativeSet = {}                             #dict representing a representative set (we choose the element in the class with the lowest values)
    for s in EquivalenceRelation:
        t = EquivalenceRelation[s]
        S = [s,t]
        smin = min(s)
        tmin = min(t)
        I = [smin, tmin]
        argmin = I.index(min(I))
        RepresentativeSet[s] = S[argmin]  
    RepresentativeSet = {v:dictNewVertices[RepresentativeSet[dictNewVerticesInv[v]]] for v in GetVerticesSimplexTree(st_sphere)}
        
    st_grass = gudhi.SimplexTree()                            #triangulation of the projective space
    for filtr in st_sphere.get_filtration():
        simplex = filtr[0]
        simplex_repr = [RepresentativeSet[v] for v in simplex]
        st_grass.insert(simplex_repr, 0)

    # Define P_map
    origin = np.ones(d+2)/(d+2)

    SimplexVectors = {}
    for i in VerticesSimplex:
        v = np.zeros(d+2)
        v[i] = 1
        SimplexVectors[i] = v

    AffineVectors = np.zeros((d+2,d+1)) 
    for i in range(d+1):
        vector = SimplexVectors[i]-origin
        AffineVectors[:,i] = vector    
    Q, r = np.linalg.qr(AffineVectors)

    ProjectedVectors = {}
    for i in range(d+2):
        vector = SimplexVectors[i]-origin[i]
        ProjectedVectors[i] = np.dot(Q.T, vector)
    
    SimplexProjections = {}
    for i in VerticesSimplex:
        vector = ProjectedVectors[i] 
        SimplexProjections[i] = VectorToProjection(vector)
        
    map_grass = {}
    for W in GetVerticesSimplexTree(st_grass):
        V = dictNewVerticesInv[W]
        A = np.zeros((d+1,d+1))
        for v in V:
            A = A + SimplexProjections[v]   
        #Project A on Grassmannian
        S = (A+A.transpose())/2
        Eigenvalues, Eigenvectors = np.linalg.eig(S)
        Eigenvalues_sort = np.argsort(Eigenvalues)
        Eigenvector = Eigenvectors[:,Eigenvalues_sort[-1]]
        Proj = np.outer(Eigenvector, Eigenvector)    
        map_grass[W] = Proj
        
    map_sphere = {}
    for V in GetVerticesSimplexTree(st_sphere):
        A = np.zeros((1,d+1))
        for v in dictNewVerticesInv[V]:
            A = A + ProjectedVectors[v]/np.linalg.norm(ProjectedVectors[v])
        map_sphere[V] =  A/np.linalg.norm(A)
        
    return st_grass, map_grass, st_sphere, map_sphere, RepresentativeSet

def GetFaceMapBundleFiltration(X, st_sphere, map_sphere, RepresentativeSet):
    '''
    Returns the minimal face map corresponding to the map map_sphere : X --> st_sphere, 
    post-composed with st_sphere --> ProjectiveSpace. 
    RepresentativeSet is a set of representative points of the relation Sphere/~ = ProjectiveSpace.
    It first computes MinimalFaceMapSphere, which is the minimal face map corresponding 
    to the map map_sphere: X --> st_sphere. It is a dict {(vertex of st_sphere):(np.array)}
    MinimalFaceMap is a dictionary {(point of X):(simplex of st_sphere)}
    
    Input:
        X (np.array): size (N x m+m**2). The input point cloud.
        st_sphere (a gudhi.SimplexTree): triangulation of the sphere.
        sphere_map (dict int:np.array): application (vertex of st_sphere)-->(np.array), 
                                        representing the coordinates of the vertices of st_sphere.
        RepresentativeSet (dict (tuple of int):(tuple of int))

    Output:
        FaceMap (dict int:(list of int)): application (vertex of S)-->(simplex of Projectivespace)
    '''
    N=np.shape(X)[0]              #number of points
    m=int((-1+np.sqrt(1+4*np.shape(X)[1]))/2) #get the integer m such that m+m**2=np.shape(X)[1]
    X_map = {i:X[i, m:m+m**2].reshape((m,m)) for i in range(N)}
    
    #Get MinimalFaceMapSphere
    FaceMapSphere = {}
    d = st_sphere.dimension()
    for v in range(N):
        v_proj = X_map[v]
        v_vect = ProjectionToVector(v_proj)

        DistancesToFaces = {}
        BarCoordinates = {}
        for face in GetSimplicesSimplexTree(st_sphere, d):
            list_vertices = [map_sphere[vertex][0] for vertex in face]

            intersect, bar_coords, distance = IntersectionLineHyperplane(v_vect, list_vertices)
            if intersect:
                if np.prod(bar_coords>=0):
                    DistancesToFaces[tuple(face)] = distance
                    BarCoordinates[tuple(face)] = bar_coords
        face_min = min(DistancesToFaces.keys(), key=(lambda k: DistancesToFaces[k])) 
            #closest face to v, among the faces such that v project onto its convex hull
        list_vertices = [map_sphere[vertex][0] for vertex in face_min]
        bar_coords = BarCoordinates[face_min] 
        FaceMapSphere[v] = [list(face_min)[i] for i in range(d+1) if bar_coords[i]>0 ]
            #the minimal face corresponds to the simplex spanned by the vertices of face_min
            #such that the projection of v onto it has nonzero barycentric coordinates
    
    #Get MinimalFaceMapProj
    FaceMap = {}
    for v in range(N):
        FaceSphere = FaceMapSphere[v]
        FaceProj = [ RepresentativeSet[vertex] for vertex in FaceSphere ]
        FaceMap[v] = FaceProj

    return FaceMap

def ComputeLifebar(st, X, filtration_max):
    '''
    Compute the lifebar of the first persistent Stiefel-Witney class of the 
    vector bundle filtration st, from the Grassmaniann st_grass and the mapping 
    cone filtration st_cone. The lifebar is computed until filtration_max.
    
    Input:
        st (gudhi.SimplexTree): a Cech simplicial filtration.
        st_grass (gudhi.SimplexTree): the Grassmaniann.
        st_cone (gudhi.SimplexTree): the mapping cone.
        filtration_max (float): the maximal value to compute the lifebar.
    
    Output: 
        Lifebar (np.array): the lifebar. Shape 1x100.
        
    Example:
        X = velour.SampleOnCircleNormalCheck(N = 30, gamma=1, sd=0)
        filtration_max = velour.BundleFiltrationMaximalValue(X, filtration='Cech')
        velour.ComputeLifebar(st, X, filtration_max)    
    '''
    d=int((-1+np.sqrt(1+4*np.shape(X)[1]))/2) #get the integer d such that d+d**2=np.shape(X)[1]
    # Triangulate the projective space
    st_grass, map_grass, st_sphere, map_sphere, RepresentativeSet = TriangulateProjectiveSpace(d-1)
    
    # Compute the face map
    FaceMap = GetFaceMapBundleFiltration(X, st_sphere, map_sphere, RepresentativeSet)

    # Try to compute a weak simplicial approximation
    g = GetWeakSimplicialApproximation(st, st_grass, FaceMap, filtration_max=filtration_max)
    if g == False:
        #the simplex tree has to be subdivided
        print('---> Subdivise again <---', flush = True)
        return False
        
    else:
        # Compute the mapping cone
        st_cone = MappingConeFiltration(st, st_grass, g, filtration_max = filtration_max)
    
        # Compute the values of the lifebar
        I = np.linspace(0,filtration_max, 100)
        dim = max([st.dimension(), st_grass.dimension(), st_cone.dimension()])
        Betti = GetBettiCurves(st, I, dim=dim)
        Betti_cone = GetBettiCurves(st_cone, I, dim=dim)
        Betti_grass = GetBettiCurves(st_grass, I, dim=dim)
        Lifebar = np.zeros((np.size(I)))
        c=1
        for d in range(1,dim-1):
            Lifebar += c*(Betti[d,:]-Betti_cone[d+1,:]+Betti_grass[d+1,:])
            c=c*(-1)    
        return Lifebar
