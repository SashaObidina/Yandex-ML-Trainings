import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps
    
    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    n = data.shape[0]
    r = np.random.rand(n).T
    
    for i in range(num_steps):
        r = data.dot(r)/np.sqrt(np.sum(np.square(data.dot(r))))
        lam = (r.T.dot(data).dot(r))/(r.T.dot(r)) 
        
    #print(lam)
    #print(r)
    
    return float(lam), r

    return 