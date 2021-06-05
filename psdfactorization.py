import numpy as np
import time
#from scipy.linalg import sqrtm

ABSERR = 10E-10


def compute_psd_factorization(X,r,nIterates=100,method='multiplicative',Init = None,silent=False):
    n1,n2 = X.shape
    
    if Init is None:
        A = gen_psdlinmap(n1,r)
        B = gen_psdlinmap(n2,r)
    else:
        A,B = Init
    
    Errs = np.zeros((nIterates,))
    
    start = time.process_time()
    
    if not(silent):
        print(' It. # | Error | Time Taken')
    
    for ii in range(nIterates):
        
        t_start = time.time()
        
        if method == 'multiplicative':
            try:
                B = update_multiplicative(A, B, X)
            except:
                print('d')
                B = update_multiplicative_damped(A, B, X)
            try:
                A = update_multiplicative(B, A, X.T)
            except:
                print('d')
                A = update_multiplicative_damped(B, A, X.T)
            
        if method == 'multiplicativeaccelerated':
            B = update_multiplicativeaccelerated(A, B, X)
            A = update_multiplicativeaccelerated(B, A, X.T)
            
        if method == 'fpgm':
            B = update_fpgm(A, B, X, 10)
            B = np.real(B)
            A = update_fpgm(B, A, X.T, 10)
            
        AB = linmap_dot(A, B)
        Errs[ii,] = np.linalg.norm(AB-X)/np.linalg.norm(X)
        
        elapsed_time = time.time() - t_start
        
        np.save('A.npy',A)
        np.save('B.npy',B)
        np.save('Errs.npy',Errs)
        
        if not(silent):
            print(str(ii+1) + ' | ' + str(Errs[ii,]) + ' | ' + str(elapsed_time))
    
    time_elapsed = time.process_time() - start
    print('Average Iteration Time: ' + str(time_elapsed/nIterates))
    print('Total Time: ' + str(time_elapsed))
    
    return {'A': A, 'B': B, 'Errors' : Errs, 'ElapsedTime' : time_elapsed}


def sqrtm(X):
    R_org = np.linalg.cholesky(X).T
    R = R_org.copy()
    ERR = 100.0
    c = 0
    while ERR > ABSERR:
        c += 1
        Rnew = 0.5 * (R + np.linalg.inv(R).T)
        ERR = np.linalg.norm(Rnew-R) / np.linalg.norm(R)
        R = Rnew
    #print(c)
    Xhalf = R.T @ R_org 
    return (Xhalf + Xhalf.T) / 2
     

def gen_psd(q):
    """
    Random PSD matrix of size (q,q)
    from Wishart distribution
    """
    
    G = np.random.randn(q,q)
    G = G @ G.T * (1/q)
    
    return G


def gen_psdlinmap(d,q):
    """
    Generate Random linear map of dimensions d,q,q
    Each "row" is a q x q random symmetric PSD matrix 
    """
    
    A = np.zeros((d,q,q))
    for i in range(d): # The rows of A are PSD matrices
        A[i,:,:] = gen_psd(q)
    
    return A


def gen_cpmap(d,q,sym=False):
    """
    Generate random linear map of dimensions d,q,q
    Each row is a q x q random symmetric matrix
    """

    # Random linear map
    A = np.random.randn(d,q,q) / np.sqrt(q*d)
    if sym==True:
        for i in range(d):
            A[i,:,:] = A[i,:,:] + A[i,:,:].T
        
    return A


def applycpmap(A,X):

    d,q,_ = A.shape
    AX = np.zeros((q,q))    

    for i in range(d):
        W = A[i,:,:]
        W = (W @ X) @ W.T
        W = (W + W.T) / 2
        AX += W
        
    return AX


def linmap_dot(A,B):
    """
    Given inputs two linear maps of size d_1,q,q and d_2,q,q
    Returns a matrix of size d1,d_2 by `contracting' these linear maps
    """
    
    d1,_,_ = A.shape
    d2,_,_ = B.shape
    
    X = np.zeros((d1,d2))
    
    for i in range(d1):
        for j in range(d2):
            X[i,j] = np.trace( A[i,:,:].T @ B[j,:,:] )
            
    return X


def applylinmap(A,X):
    """
    Input A is a linear map of size d,q,q
    and X is a matrix of size q,q
    
    Output is A(X) a vector of size d
    """
    
    d,q,_ = A.shape
    AX = np.zeros((d,))
    
    for i in range(d):
        AX[i,] = np.trace(A[i,:,:].T @ X)
        
    return AX


def applytransposelinmap(A,x):
    """
    Input A is a linear map of size d,q,q
    and x is a vector of size d
    
    Output is A^T x a matrix of size q,q
    """
    
    d,q,_ = A.shape
    AtX = np.zeros((q,q))
    
    for i in range(d):
        AtX += A[i,:,:] * x[i,]
        
    return AtX


def applycompositionoperator(A,X):
    """
    Input A is a linear map of size d,q,q while X is a matrix of size q,q
    Output is A^T A(X), which is a matrix of size q,q
    """
    
    return applytransposelinmap(A, applylinmap(A, X))






def vectorizelinmap(A):
    """
    Takes as input a linear map of size d,q_1,q_2
    Outputs a linear map of size d,q_1*q_2 where the second and third dimensions
    are flattened into a vector
    """
    
    d,q_1,q_2 = A.shape
    Aflat = np.zeros((d,q_1*q_2))
    for i in range(d):
        Aflat[i,:] = np.matrix.flatten(A[i,:,:])

    return Aflat


def reversevectorizelinmap(A,q_1,q_2):
    """
    Reverse of vectorize linear map
    """
    
    d,_ = A.shape
    Afull = np.zeros((d,q_1,q_2))
    for i in range(d):
        Afull[i,:,:] = np.reshape(A[i,:],(q_1,q_2))

    return Afull


def matrix_gm(A,B):
    """
    Given as inputs two positive definite matrices A and B
    Returns the Geometric Mean
    """
    Ahalf = sqrtm(A)
    Ainvminus = np.linalg.inv(Ahalf)
    
    G = (Ainvminus @ B) @ Ainvminus.T
    G = (G + G.T) / 2.0
    G = sqrtm(G)
    
    gm = (Ahalf @ G) @ Ahalf.T
    gm = (gm + gm.T) / 2
    
    return gm


def vanloan_tx(T,q):
    S = np.zeros(T.shape)
    for i in range(q):
        for j in range(q):
            for x in range(q):
                for y in range(q):
                    v = T[i+x*q,j+y*q]
                    s = i + j*q
                    t = x + y*q
                    S[s,t] = v
    return S


def projPSD(X):
    X = (X + X.T) / 2
    D,U = np.linalg.eig(X)
    D = np.maximum(D,np.zeros(D.shape))
    Z = ( U @ np.diag(D) ) @ U.T
    Z = np.real(Z)
    return Z


def _isPSD(X):
    D,U = np.linalg.eig(X)
    if min(D) < 0.0:
        return False
    else:
        return True


def update_multiplicative(A,B,X):
    """
    Updates the factor B via the multiplicative updates
    """

    B_new = np.zeros(B.shape)
    d,q,_ = B.shape
    
    for i in range(d):
        AAB = applytransposelinmap(A,applylinmap(A,B[i,:,:])) 
        AtX = applytransposelinmap(A,X[:,i])
        AABinv = np.linalg.inv(AAB)
        
        # Conditioning
        E = np.identity(q) * 10E-8 
        
        H = matrix_gm(AABinv+E,B[i,:,:]+E) 
        B_new[i,:,:] = ( H @ AtX ) @ H.T
        
    return B_new


def update_multiplicative_damped(A,B,X):
    """
    Updates the factor B via the multiplicative updates
    """

    B_new = np.zeros(B.shape)
    d,q,_ = B.shape
    
    for i in range(d):
        AAB = applytransposelinmap(A,applylinmap(A,B[i,:,:])) 
        AtX = applytransposelinmap(A,X[:,i])
        AABinv = np.linalg.inv(AAB)
        
        # Conditioning
        E = np.identity(q) * 10E-8 
        
        H = matrix_gm(AABinv+E,B[i,:,:]+E) 
        B_new[i,:,:] = ( H @ AtX ) @ H.T
        
    return B_new


def update_multiplicativeaccelerated(A,B,X):
    """
    Updates the factor B via the multiplicative updates
    """

    d,_,_ = B.shape
    
    for i in range(d):
        AAB = applytransposelinmap(A,applylinmap(A,B[i,:,:])) 
        AtX = applytransposelinmap(A,X[:,i])
        AABinv = np.linalg.inv(AAB)
        
        Q = AtX - AAB
        H = matrix_gm(AABinv,B[i,:,:])
        P = H @ Q @ H.T
        
        c1 = np.trace(P.T @ Q)
        c2 = np.linalg.norm(applylinmap(A,P))**2
        Bh = sqrtm(np.linalg.inv(B[i,:,:]))
        XPX = Bh @ P @ Bh.T
        D,_ = np.linalg.eig(XPX)
        c3 = np.min(D)
        c4 =  - 0.9 / c3
        
        
        c = min(c1/c2,c4)
        
        if c < 0:
            print('Negative')
        
        B[i,:,:] += c * P
        
        if _isPSD(B[i,:,:]) == False:
            print(c - c4)
            print('Error!!!')
        
    return B


def update_fpgm(A,B,X,nIterates):
    
    Af = vectorizelinmap(A)
    AA = Af.T @ Af
    D,_ = np.linalg.eig(AA)
    L = 2*np.max(np.abs(D))
    B_0 = B.copy()
    B_1 = B.copy()
    
    d,q,_ = B.shape
    
    for t in range(nIterates):
        factor = (t-2) / (t+1)
        Y = B_1 + factor * (B_1 - B_0)
        
        XA = np.zeros(A.shape)
        for j in range(d):
            XA[j,:,:] = applytransposelinmap(A,X[:,j])
        Yf = vectorizelinmap(Y)
        #print(Yf.shape)
        YAA = reversevectorizelinmap(Yf @ AA,q,q)

        B_2 = Y + (XA - YAA)/L
        for j in range(d):
            #print(np.linalg.norm(B_2[j,:,:] - B_2[j,:,:].T))
            B_2[j,:,:] = projPSD(B_2[j,:,:])
            
        B_0 = B_1.copy()
        B_1 = np.real(B_2.copy())
        
        #AB = linmap_dot(A,B_2)
        #print(np.linalg.norm(X-AB))
        
    return B_1
        

### Codes for non-embeded block matrices

def compute_blockpsd_factorization(X,blockpattern,nIterates=100,method='multiplicative',Init = None,silent=False):
    n1,n2 = X.shape
    
    if Init is None:
        A = gen_blockpsdlinmap(n1,blockpattern)
        B = gen_blockpsdlinmap(n2,blockpattern)
    else:
        A,B = Init
    
    Errs = np.zeros((nIterates,))
    
    start = time.process_time()
    
    if not(silent):
        print(' It. # | Error | Time Taken')
    
    for ii in range(nIterates):
        
        t_start = time.time()
        
        """
        if method == 'multiplicative':
            try:
                B = update_blockmultiplicative(A, B, X)
            except:
                print('d')
                B = update_multiplicative_damped(A, B, X)
            try:
                A = update_blockmultiplicative(B, A, X.T)
            except:
                print('d')
                A = update_multiplicative_damped(B, A, X.T)
        """
        B = update_blockmultiplicative(A, B, X)
        A = update_blockmultiplicative(B, A, X.T)
        
        AB = blocklinmap_dot(A, B)
        Errs[ii,] = np.linalg.norm(AB-X)/np.linalg.norm(X)
        
        elapsed_time = time.time() - t_start
        
        np.save('A.npy',A)
        np.save('B.npy',B)
        np.save('Errs.npy',Errs)
        
        if not(silent):
            print(str(ii+1) + ' | ' + str(Errs[ii,]) + ' | ' + str(elapsed_time))
    
    time_elapsed = time.process_time() - start
    print('Average Iteration Time: ' + str(time_elapsed/nIterates))
    print('Total Time: ' + str(time_elapsed))
    
    return {'A': A, 'B': B, 'Errors' : Errs, 'ElapsedTime' : time_elapsed}


def update_blockmultiplicative(A,B,X):
    """
    Updates the factor B via the multiplicative updates
    """
    
    l = len(B)
    d = B[0].shape[0]
    
    AAB = compute_blockAAB(A,B)
    AtX = compute_block(A,X)
    
    # Invert AAB
    AABinv = []
    for k in range(l):
        AABinv.append(np.zeros(B[k].shape))
        for i in range(d):
            AABinv[k][i,:,:] = np.linalg.inv(AAB[k][i,:,:])
    
    # Compute the geometric mean with a damping term
    H = []
    for k in range(l):
        H.append(np.zeros(B[k].shape))
        for i in range(d):
            H[k][i,:,:] = matrix_gm(AABinv[k][i,:,:],B[k][i,:,:])
            
    # Update B_new
    B_new = []
    for k in range(l):
        B_new.append(np.zeros(B[k].shape))
        for i in range(d):
            M = ( H[k][i,:,:] @ AtX[k][i,:,:] ) @ H[k][i,:,:].T
            B_new[k][i,:,:] = M
            
    return B_new


def compute_blockAAB(A,B):
    m = A[0].shape[0]
    n = B[0].shape[0]
    X = np.zeros((m,n))
    l = len(A)
    
    for ii in range(m):
        for jj in range(n):
            for k in range(l):
                X[ii,jj] += np.trace( A[k][ii,:,:].T @ B[k][jj,:,:] )
    
    # For the last step, implement compute_block
    AAB = compute_block(A,X)    
    
    return AAB


def compute_block(A,X):
    """
    Computes A^T X for block diagonal linear maps A
    """
    l = len(A)
    m,n = X.shape
    
    AX = []
    
    # Initialize
    for k in range(l):
        q = A[k].shape[1]
        AX.append(np.zeros((n,q,q)))
        
    # Product
    for k in range(l):
        for ii in range(m):
            for jj in range(n):
                AX[k][jj,:,:] += X[ii,jj]*A[k][ii,:,:]
                
    return AX

def gen_blockpsdlinmap(d,blockpattern):
    """
    Generate Random linear map, with block patterns 
    """
    A = []
    for j in range(len(blockpattern)):
        q = blockpattern[j]
        A.append(gen_psdlinmap(d,q))
    return A


def blocklinmap_dot(A,B):
    """
    Same as linmap_dot, but for block diagonals
    """
    
    d1,_,_ = A[0].shape
    d2,_,_ = B[0].shape
    l = len(A)
    
    X = np.zeros((d1,d2))
    
    for i in range(d1):
        for j in range(d2):
            for k in range(l):
                X[i,j] += np.trace( A[k][i,:,:].T @ B[k][j,:,:] )
            
    return X
    

### Code for embeded block

def compute_block2psd_factorization(X,blockpattern,nIterates=100,method='multiplicative',Init = None,silent=False):
    n1,n2 = X.shape
    
    if Init is None:
        A = gen_embedblockpsdlinmap(n1,blockpattern)
        B = gen_embedblockpsdlinmap(n2,blockpattern)
    else:
        A,B = Init
    
    Errs = np.zeros((nIterates,))
    
    start = time.process_time()
    
    if not(silent):
        print(' It. # | Error | Time Taken')
    
    for ii in range(nIterates):
        
        t_start = time.time()
        
        #B = update_embedblockmultiplicative(A, B, X, blockpattern)
        #A = update_embedblockmultiplicative(B, A, X.T, blockpattern)
        B = update_multiplicative(A, B, X)
        A = update_multiplicative(B, A, X.T)
        
        AB = linmap_dot(A, B)
        Errs[ii,] = np.linalg.norm(AB-X)/np.linalg.norm(X)
        
        elapsed_time = time.time() - t_start
        
        np.save('A.npy',A)
        np.save('B.npy',B)
        np.save('Errs.npy',Errs)
        
        if not(silent):
            print(str(ii+1) + ' | ' + str(Errs[ii,]) + ' | ' + str(elapsed_time))
    
    time_elapsed = time.process_time() - start
    print('Average Iteration Time: ' + str(time_elapsed/nIterates))
    print('Total Time: ' + str(time_elapsed))
    
    return {'A': A, 'B': B, 'Errors' : Errs, 'ElapsedTime' : time_elapsed}


def gen_embedblockpsdlinmap(d,blockpattern):
    """
    Generate Random linear map, matrix has block patterns
    """
    q = sum(i for i in blockpattern) # Length of block pattern
    l = len(blockpattern)
        
    A = np.zeros((d,q,q))
    c = 0
    for j in range(l): # The rows of A are PSD matrices
        qs = blockpattern[j]
        for i in range(d):
            A[i,c:c+qs,c:c+qs] = gen_psd(qs)
        c += qs
    return A


def _breakintoblocklist(X,blockpattern):
    Xlist = []
    c = 0
    for j in range(len(blockpattern)):
        qs = blockpattern[j]
        Xlist.append(X[c:c+qs,c:c+qs])
        c += qs
    return Xlist
    
    

def update_embedblockmultiplicative2(A,B,X,blockpattern):
    """
    Updates the factor B via the multiplicative updates
    """

    B_new = np.zeros(B.shape)
    d,q,_ = B.shape
    l = len(blockpattern)
    
    for i in range(d):
        AAB = applytransposelinmap(A,applylinmap(A,B[i,:,:])) 
        AtX = applytransposelinmap(A,X[:,i])
        
        c = 0
        AABinv = np.zeros(AAB.shape)
        for j in range(l):
            qs = blockpattern[j]
            AABinv[c:c+qs,c:c+qs] = np.linalg.inv(AAB[c:c+qs,c:c+qs])
            c += qs
            
        # Conditioning
        E = np.identity(q) * 10E-8 
        
        c = 0
        for j in range(l):
            qs = blockpattern[j]
            H = matrix_gm(AABinv[c:c+qs,c:c+qs]+E[c:c+qs,c:c+qs],B[i,c:c+qs,c:c+qs]+E[c:c+qs,c:c+qs])
            B_new[i,c:c+qs,c:c+qs] = ( H @ AtX[c:c+qs,c:c+qs] ) @ H.T
            c += qs
        
    return B_new


def update_embedblockmultiplicative(A,B,X,blockpattern):
    """
    Updates the factor B via the multiplicative updates
    """

    B_new = np.zeros(B.shape)
    d,q,_ = B.shape
    l = len(blockpattern)
    
    for i in range(d):
        Blist = _breakintoblocklist(B[i,:,:],blockpattern)
        AAB = applytransposelinmap(A,applylinmap(A,B[i,:,:])) 
        AtX = applytransposelinmap(A,X[:,i])
        AAB = _breakintoblocklist(AAB,blockpattern)
        AtX = _breakintoblocklist(AtX,blockpattern)
        
        AABinv = [np.linalg.inv(j) for j in AAB]
            
        # Conditioning
        E = _breakintoblocklist(np.identity(q) * 10E-8,blockpattern)
        
        H = [matrix_gm(AABinv[j]+E[j],Blist[j]+E[j]) for j in range(l)]
        B_new_list = [ (H[j] @ AtX[j]) @ H[j].T for j in range(l) ]
        c = 0
        for j in range(l):
            qs = blockpattern[j]
            B_new[i,c:c+qs,c:c+qs] = B_new_list[j]
            c += qs
        
    return B_new