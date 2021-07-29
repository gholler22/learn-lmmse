import numpy as np


class InverseProblem():
    '''Abstract base class for InverseProblem.
    
    Attributes:
        dim_X : int
            dimension of the prior X
        dim_Y : int
            dimension of the data Y
        A : ndarray of shape (dim_Y, dim_X)  
            forward operator
        cov_Y : ndarray of shape (dim_Y, dim_Y) 
            covariance of the data
        LMMSE : ndarray of shape (dim_Y, dim_X)
            transposed linear minimum mean squared error (LMMSE) estimator
        cov_E : ndarray of shape (dim_Y, dim_Y)
            covariance of the error of the LMMSE estimator
        MSE   : float
            mean squared error of the LMMSE estimator
    '''
    def __init__(self, forward_operator=None, dim_X=None, dim_Y=None, 
                 seed=None):
        '''Default class initializer.
        
        Initializes the attributes A, dim_X, and dim_Y.
        
        Parameters:
            forward_operator : ndarray of shape (dim_Y, dim_X) or string
                2D array or string specifying the forward operator.
                Available options for string: 
                    -'random': matrix with independent standard normal entries
                    -'identity': identity matrix
            dim_X : int
                dimension of the prior X. To be used if and only if 
                forward_operator is a string.  
            dim_Y: int, optional
                dimension of the data Y. To be used only if 
                forward_operator=='random'.
            seed: int, optional
                Seed used for creation of the 'random' forward operator. 
                To be used only if forward_operator=='random'.
        '''
        if not isinstance(forward_operator, str):
            try:
                assert dim_X is None and dim_Y is None
            except AssertionError:
                print('dim_X and dim_Y must be None' 
                      + 'when forward_operater is provided:')
                raise
            self.A = forward_operator
        elif forward_operator == 'random':
            if dim_Y is None: dim_Y = dim_X
            rng2 = np.random.RandomState(seed=seed)
            self.A = rng2.randn(dim_Y, dim_X)
        elif forward_operator == 'identity':
            try:
                assert dim_X == dim_Y or dim_Y is None
            except AssertionError:
                print('dim_X and dim_Y must be equal for identity:')
                raise
            self.A = np.eye(dim_X)
            
        self.dim_X = self.A.shape[1]
        self.dim_Y = self.A.shape[0]
        
    def create_samples(self, n_samples, seed=None, sample_type='train'):
        '''Creates random training or test samples of prior-data-noise triples.
        
        Parameters:
            n_samples : int
                number of samples to be generated
            seed : int, optional
                seed to be used for random number generation 
            sample_type: string, optional
                Choose sample_type='train' to create training samples and
                sample_type='test' to create test samples. Used only in some
                subclasses.
                
        Returns:
            X : ndarray of shape (n_samples, dim_X)
                2D array containing the samples of the unknown as rows
            Y : ndarray of shape (n_samples, dim_Y)
                2D array containing the samples of the data as rows
            Z : ndarray of shape (n_samples, dim_Y)
                2D array containing the samples of the noise as rows
        '''
        pass
    

class GaussianInverseProblem(InverseProblem):
    '''Class of a linear inverse problem where prior and noise are Gaussian.
    
    This creates a Gaussian Inverse Problem with a randomly chosen covariances
    for prior and noise.
    
    Attributes: 
        sqrt_cov_X: ndarray of shape (dim_X, dim_X)
            square root of covariance matrix of the prior
        sqrt_cov_Z: ndarray of shape (dim_Y, dim_Y)
            square root of covariance matrix of the noise
        For the other attributes see the parent class InverseProblem.
    '''
    def __init__(self, forward_operator=None, dim_X=None, dim_Y=None, 
                 seed=None):
        super().__init__(forward_operator=forward_operator, dim_X=dim_X, 
                         dim_Y=dim_Y, seed=seed)

        if seed is None:
            seed1 = None
            seed2 = None
        else:
            seed1 = seed+1
            seed2 = seed+2
        
        rng0 = np.random.RandomState(seed=seed1)
        eig_cov_X = rng0.rand(self.dim_X)
        rows_x = rng0.randn(self.dim_X, self.dim_X)
        qx, rx = np.linalg.qr(rows_x)
        qx = qx * np.sign(np.diag(rx))
        self.sqrt_cov_X = qx @ (qx * np.sqrt(eig_cov_X)).T
        cov_X = np.linalg.matrix_power(self.sqrt_cov_X, 2) 

        rng1 = np.random.RandomState(seed=seed2)
        eig_cov_Z = rng1.rand(self.dim_Y)
        rows_z = rng1.randn(self.dim_Y, self.dim_Y)
        qz, rz = np.linalg.qr(rows_z)
        qz = qz * np.sign(np.diag(rz))
        self.sqrt_cov_Z = qz @ (qz * np.sqrt(eig_cov_Z)).T
        cov_Z = np.linalg.matrix_power(self.sqrt_cov_Z, 2)

        self.cov_Y = self.A @ cov_X @ self.A.T + cov_Z
        self.LMMSE = np.linalg.solve(self.cov_Y, self.A @ cov_X)
        self.cov_E = cov_X - cov_X @ self.A.T @ self.LMMSE 
        self.MSE = np.trace(self.cov_E)

    def create_samples(self, n_samples=2, seed=None, sample_type='None'):
        rng = np.random.RandomState(seed)
        aux = rng.randn(n_samples, self.dim_X+self.dim_Y)
        X = aux[:,:self.dim_X] @ self.sqrt_cov_X
        Z = aux[:,self.dim_X:self.dim_X+self.dim_Y] @ self.sqrt_cov_Z
        Y = X @ self.A.T + Z
        return X,Y,Z


class SampledInverseProblem(InverseProblem):
    '''Linear inverse problem with prior from some database. 
    
    This creates a linear inverse problem where the prior is observed 
    only through samples from some provided database.
    
    Attributes:
        X_train: ndarray of shape (#train_samples, dim_X)
            ground truth training images 
        X_test: ndarray of shape (#test_samples, dim_X)
            ground truth test images
        noiselevel: float
            The noiselevel is such that the covariance of the noise Z is 
            noiselevel**2 identity.
        noise_type: string
            Specifies the noise type. Available: 'Gaussian' and 'uniform'.
        For the other attributes see the parent class InverseProblem.
    '''
    def __init__(self, database, noiselevel=0.1, noise_type='Gaussian'):
        '''Default class initializer.
        
        Parameters:
            database : function()
                Function such that "X_train, X_test = database()" returns 
                ndarrays of shapes (#training_samples, dim_X) and 
                (#test_samples, dim_X).
            noiselevel : float
                The noiselevel is such that the covariance of the noise Z is 
                noiselevel**2 identity.
            noise_type : string
                Specifies the noise type. Available: 'Gaussian' and 'uniform'
        '''
        X_train, X_test = database()
        super().__init__(forward_operator='identity', dim_X=X_train.shape[1])

        self.cov_X = np.cov(X_train, rowvar = False)  
        self.X_train = X_train - np.mean(X_train, axis = 0)        
        self.X_test = X_test - np.mean(X_test, axis = 0)
        cov_Z = noiselevel**2 * np.identity(self.X_train.shape[1])
        self.noiselevel = noiselevel
        self.cov_Y = self.A @ self.cov_X @ self.A.T + cov_Z
        self.LMMSE = np.linalg.solve(self.cov_Y, self.A @ self.cov_X)
        self.cov_E = self.cov_X - self.cov_X @ self.A.T @ self.LMMSE 
        self.noise_type = noise_type
        self.MSE = np.trace(self.cov_E)

    def create_samples(self, n_samples, seed=None, sample_type='train'):
        rng = np.random.RandomState(seed=seed)
        if sample_type == 'train':
            chosen_samples = rng.choice(self.X_train.shape[0], n_samples, 
                                        replace=False)
            X = self.X_train[chosen_samples,:]
        if sample_type == 'test':
            chosen_samples = rng.choice(self.X_test.shape[0], n_samples, 
                                        replace=False)
            X = self.X_test[chosen_samples,:]

        if self.noise_type == 'Gaussian':
            Z = self.noiselevel * rng.standard_normal(X.shape)
        elif self.noise_type == 'uniform':
            Z = self.noiselevel * rng.uniform(-np.sqrt(3), np.sqrt(3), 
                                              size=X.shape)
        Y = X @ self.A.T + Z
        return X,Y,Z        