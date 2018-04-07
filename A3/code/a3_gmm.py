from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
import scipy
#np.set_printoptions(threshold=np.inf)

dataDir = '/u/cs401/A3/data/'
#dataDir = '../data/' # TODO: change this back before submission

class theta:
    def __init__(self, name, M=8,d=13):
        self.name = name
        self.omega = np.zeros((M,1)) # weights of each m GMM: initialize randomly, sum should equal 1 at all time cause it's the prob of a GMM
        self.mu = np.zeros((M,d))  # means: should init with actual random MFCC vector from each speaker
        self.Sigma = np.zeros((M,d))  # should init this with identity matrix, do it before passing to in train, before passing to bmx etc


def log_b_m_x( m, x, myTheta, preComputedForM=None):  # preComputedForM can be a numpy array, changed signature to reflect this
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM

        m (int): index of gaussian mixture component (0 to M-1)
        x ((T, d) numpy array): d-dimensional vector
        myTheta (theta): contains omega/mu/Sigma for this m_th component
        preComputedForM (list of floats): list of precomputed term for each m
    '''

    M = np.shape(myTheta.omega)[0]
    b = np.zeros(M)#0
    d = np.shape(myTheta.mu)[1]

    T = np.shape(x)[0]  # x is of shape (10000+, 13)

    if not preComputedForM:

        # try to use the tut slide formula insteadof eqn 1 of handout
        first_term = np.zeros(T)
        second_term = (d / 2.0) * np.log(2 * np.pi)
        third_term = np.ones(T)

        for n in range(0, d):  # up to, including d -> actually, it should be indexed from 0

            if (np.shape(x)[0] == d):
                first_term += ((x[n] - myTheta.mu[m][n]) ** 2) * ((2 * myTheta.Sigma[m][n]) ** -1)
                third_term *= myTheta.Sigma[m][n]
            else:
                first_term += ((x[:,n] - myTheta.mu[m][n]) ** 2) * ((2 * myTheta.Sigma[m][n]) ** -1)
                third_term *= myTheta.Sigma[m][n]
          

        third_term = 0.5 * np.log(third_term)
        b = -first_term - second_term - third_term  # in log e

    else:

        print("precomputation for this m exists")

        # we then assume that in the m_th position of preComputedForM contains the second half computation corresponding to this m
        try:
            second_half = preComputedForM[m]  # for the second half of the long equation from the tutorial slides independent of x
        except IndexError:
            print("preComputedForM does not contain precomputation for this m")

        first_half = 0

        for n in range(0, d):  # up to, including d -> actually, it should be indexed from 0
            first_half += 0.5 * (x[n] ** 2) * (myTheta.Sigma[m][n] ** -1) - (myTheta.mu[m][n] * x[n] * (myTheta.Sigma[m][n] ** -1))

        first_half = -first_half # add minus in front
        b = first_half - second_half # make sure to not compute second half with that minus sign in front

    # return either a single log probability for x
    # or a Tx1 array of log probabilities for X, each entry corresponding to frame t=1..T
    # which it returns will be based on the dimensions of x

    return b
    

def log_p_m_x( m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout

        m (int): index of gaussian mixture component (0 to M-1)
        x ((T, d) numpy array): d-dimensional vector
        myTheta (theta): contains omega/mu/Sigma for this m_th component
    '''


    # how would we pass in precomputation for m if we have to call this function here...
    numerator = np.log(myTheta.omega[m]) + log_b_m_x(m, x, myTheta)

    M = np.shape(myTheta.omega)[0]

    #a = np.zeros(M)  # stores all our b_k(x_t) for k=1..M, which are in log
    c = np.zeros(np.shape(x)[0])
    for k in range(M):

        c += scipy.special.logsumexp(a=log_b_m_x(k, x, myTheta), b=myTheta.omega[k])
        # If b is given then np.log(np.sum(b*np.exp(a))) is returned.

    denominator = c

    # for comparison
    #print(np.exp(numerator) / np.exp(denominator))  # this actually gives us [0. ], prob cause of underflow, tis why we should keep things in base e as long as possible
    
    # finally we can subtract in base e to perform division in non-e
    return numerator - denominator  # cause it's still in base e
    # btw, our return is a single valued vector [v_0], we can easily extend this to returning a larger vector

    
def logLik( log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency. 


        See equation 3 of the handout
    '''
    # log_Bs a list of log probabilities of x_t in different components m

    # get the b_m(x_t) via log_Bs[m][t] -> a scalar (for vectorization, we can use log_Bs[m] -> precomputation for t=1..T for m)

    M = np.shape(log_Bs)[0]
    T = np.shape(log_Bs)[1]
    # non vectorized, gonna need nested for loops
    P = 0
    P_arr = np.zeros(T)
    for t in range(T):

        arr = np.zeros(M)
        for m in range(M):
            arr[m] = np.log(myTheta.omega[m]) + log_Bs[m][t]
        P += scipy.special.logsumexp(arr)  # adding log p(x_t, theta_s)

    # vectorized, only outer for loop

    return P

    
def train( speaker, X, M=8, epsilon=0.0, maxIter=20 ):
    ''' 
        Train a model for the given speaker. Returns the theta (omega, mu, sigma)

        speaker (string): speaker's identifying name
        X ((N, d) numpy array): data matrix, N is the number of frames/rows, d the data dimension (13)
        M (int): number of GMMs
        epsilon (float): threshold for our convergence
        maxIter (int): max number of iterations before breaking
    '''


    ### 1. Initialize theta for this speaker ###
    myTheta = theta( speaker, M, X.shape[1] )

    N = np.shape(X)[0]  # N=T=num of frames/rows of X
    d = np.shape(X)[1]  # data dimension of X

    myTheta.omega = np.random.rand(M, 1)  # initialize with random values
    myTheta.omega /= np.sum(myTheta.omega)  # divide matrix by sum of its values to make sure new sum of values equal 1

    # init mu to random MFCC vector from X
    for m in range(M):
        myTheta.mu[m] = X[random.randint(0, N)]
    
    myTheta.Sigma = np.ones((M, d))


    iteration = 0

    ### For either maxIter steps, or until convergence, repeat the following 2 steps ###
    while True:
        print("Iteration %d" % (iteration + 1))

        ### 2. Expectation ###
        log_Bs = np.zeros((M, N))  # N = T = num of frames = num of rows of X
        log_pmx = np.zeros((M, N))
        for m in range(M):
            log_Bs[m, ] = log_b_m_x(m, X, myTheta)
            log_pmx[m, ] = log_p_m_x(m, X, myTheta)
            

        ### 3. Maximization: update our theta ###
        prevTheta = myTheta  # we will always set the prev theta to the myTheta before updating it
        # we need this to compare log P(X|theta_i+1) - log P(X|theta_i) < epsilon

        # note: we are supposed to exponentiate pmx in the updating part, ie below...

        for m in range(M):
            new_omega = 0

            new_mu = np.zeros((1, d))
            new_mu_denom = 0

            new_sigma_col = np.zeros((1, d))
            new_sigma_col_denom = 0

            for t in range(N):

                new_omega += np.exp(log_pmx[m, t])

                new_mu += np.exp(log_pmx[m, t]) * X[t, ] 
                new_mu_denom += np.exp(log_pmx[m, t])

                new_sigma_col += np.exp(log_pmx[m, t]) * (X[t, ] ** 2)
                new_sigma_col_denom += np.exp(log_pmx[m, t])

            new_omega /= N
            myTheta.omega[m] = new_omega

            new_mu /= new_mu_denom
            myTheta.mu[m, ] = new_mu

            new_sigma_col /= new_sigma_col_denom
            new_sigma_col -= (new_mu ** 2)
            myTheta.Sigma[m, ] = new_sigma_col


        iteration += 1
        if (iteration >= maxIter) or (logLik(log_Bs, myTheta) - logLik(log_Bs, prevTheta) < epsilon):
            break

    return myTheta


def test( mfcc, correctID, models, k=5 ):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK] 

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    bestModel = -1
    print ('TODO')
    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    print('TODO: you will need to modify this main block for Sec 2.3')
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print( speaker )

            files = fnmatch.filter(os.listdir( os.path.join( dataDir, speaker ) ), '*npy')
            random.shuffle( files )
            
            testMFCC = np.load( os.path.join( dataDir, speaker, files.pop() ) )
            testMFCCs.append( testMFCC )

            X = np.empty((0,d))
            for file in files:
                myMFCC = np.load( os.path.join( dataDir, speaker, file ) )
                X = np.append( X, myMFCC, axis=0)

            trainThetas.append( train(speaker, X, M, epsilon, maxIter) )

    # evaluate 
    numCorrect = 0;
    for i in range(0,len(testMFCCs)):
        numCorrect += test( testMFCCs[i], i, trainThetas, k ) 
    accuracy = 1.0*numCorrect/len(testMFCCs)

    # my test
    print("Accuracy %.2f" % accuracy)
