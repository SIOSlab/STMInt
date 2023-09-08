from sympy import *
import numpy as np
from scipy.integrate import solve_ivp
import math
from astropy import constants as const
from astropy import units as u
from scipy.linalg import eigh, norm, svd
import string
import itertools
import functools
from STMint import STMint


# ======================================================================================================================
# Nonlinearity Index Functions
# ======================================================================================================================

def nonlin_index_inf_2(self, stm, stt):
    """ Function to calculate the nonlinearity index

   The induced infinity-2 norm is used in this calculation

    Args:
        stm (np array)
            State transition matrix

        stt (np array)
            Second order state transition tensor

    Returns:
        nonlinearity_index (float)
    """
    sttNorm = 0
    stmNorm = 0
    for i in range(len(stm)):
        w = eigh(stt[i, :, :], eigvals_only=True)
        sttNorm = max(sttNorm, abs(max(w, key=abs)))
        rowNorm = norm(stm[i, :])
        stmNorm = max(stmNorm, rowNorm)
    return sttNorm / stmNorm


def nonlin_index_unfold(self, stm, stt):
    """ Function to calculate the nonlinearity index

   The induced 2 norm of the unfolded STT is used in this calculation

    Args:
        stm (np array)
            State transition matrix

        stt (np array)
            Second order state transition tensor

    Returns:
        nonlinearity_index (float)
    """
    dim = len(stm)
    sttNorm = norm(np.reshape(stt, (dim, dim ** 2)), 2)
    stmNorm = norm(stm, 2)
    return sttNorm / stmNorm


def nonlin_index_frob(self, stm, stt):
    """ Function to calculate the nonlinearity index

   The frobenius norm of the STT is used in this calculation

    Args:
        stm (np array)
            State transition matrix

        stt (np array)
            Second order state transition tensor

    Returns:
        nonlinearity_index (float)
    """
    dim = len(stm)
    sttNorm = norm(np.reshape(stt, (dim, dim ** 2)), "fro")
    stmNorm = norm(stm, "fro")
    return sttNorm / stmNorm


def nonlin_index_2(self, stm, stt):
    """ Function to calculate the nonlinearity index

    An approximation of the induced 2 norm of the STT is used in this calculation
    One iteration of singular value decomposition of the contracted STT is taken
    with the maximal right singular vector of the STM as an initial guess.

    Args:
        stm (np array)
            State transition matrix

        stt (np array)
            Second order state transition tensor

    Returns:
        nonlinearity_index (float)
    """
    _, _, vh = svd(stm)
    stmVVec = vh[0, :]
    _, _, vh1 = svd(np.einsum('ijk,k->ij', stt, stmVVec))
    stt_vec = vh1[0, :]
    sttNorm = norm(np.einsum('ijk,j,k->i', stt, stt_vec, stt_vec), 2)
    stmNorm = norm(stm, 2)
    return sttNorm / stmNorm


# ======================================================================================================================
# Power Iteration Functions
# ======================================================================================================================

def power_iterate_string(self, tens):
    """ Function to calculate the index string for einsum (up to 26 dimensional tensor)

    Args:
        tens (np array)
            Tensor

    Returns:
        einsum string to perform power iteration (string)
    """
    assert (tens.ndim <= 26)
    # looks like "zabcd,a,b,c,d->z"
    stringEin = "z"
    stringContract = string.ascii_lowercase[:tens.ndim - 1]
    secondString = ""
    for char in stringContract:
        secondString += "," + char
    stringEin += stringContract + secondString + "->" "z"
    return stringEin


def tensor_square_string(self, tens):
    """ Function to calculate the index string for einsum (up to 1-13 dimensional tensor)
    Args:
        tens (np array)
            Tensor

    Returns:
        einsum string to perform tensor squaring (string)
    """
    assert (tens.ndim < 13)
    # looks like "abcd,azyx-bcdzyx>"
    firstString = string.ascii_lowercase[1:tens.ndim]
    secondString = string.ascii_lowercase[26:26 - tens.ndim:-1]
    stringEin = "a" + firstString + ",a" + secondString + "->" + firstString + secondString
    return stringEin


def power_iterate(self, stringEin, tensOrder, tens, vec):
    """ Function to perform one higher order power iteration on a symmetric tensor

    Single step

    Args:
        stringEin (string)
            String to instruct einsum to perform contractions

        tensOrder (int)
            Order of the tensor

        tens (np array)
            Tensor

        vec (np array)
            Vector

    Returns:
        vecNew (np array)

        vecNorm (float)
    """
    vecNew = np.einsum(stringEin, tens, *([vec] * (tensOrder - 1)))
    vecNorm = np.linalg.norm(vecNew)
    return vecNew / vecNorm, vecNorm


def power_iteration(self, tens, vecGuess, maxIter, tol):
    """ Function to perform higher order power iteration on a symmetric tensor

    Args:
        tens (np array)
            Tensor

        vec (np array)
            Vector

        maxIter (int)
            Max number of iterations to perform

        tol (float)
            Tolerance for difference and iterates

    Returns:
        eigVec (np array)

        eigValue (np array)
    """
    stringEin = self.power_iterate_string(tens)
    tensOrder = tens.ndim
    vec = None
    vecNorm = None
    for i in range(maxIter):
        vecPrev = vecGuess
        vec, vecNorm = self.power_iterate(stringEin, tensOrder, tens, vecPrev)
        if np.linalg.norm(vec - vecPrev) < tol:
            break
    return vec, vecNorm


def symmetrize_tensor(self, tens):
    """ Symmetrize a tensor

    Args:
        tens (np array)
            Tensor

    Returns:
        symTens (np array)
    """
    dim = tens.ndim
    rangedim = range(dim)
    tensDiv = tens / math.factorial(dim)
    permutes = map(lambda sigma: np.moveaxis(tensDiv, rangedim, sigma), itertools.permutations(range(dim)))
    symTens = functools.reduce(lambda x, y: x + y, permutes)
    return symTens


def power_iterate_symmetrizing(self, stringEin, tensOrder, tens, vec):
    """ Function to perform one higher order power iteration on a non-symmetric tensor

    Args:
        stringEin (string)
            String to instruct einsum to perform contractions

        tensOrder (int)
            Order of the tensor

        tens (np array)
            Tensor

        vec (np array)
            Vector

    Returns:
        vecNew (np array)

        vecNorm (float)
    """
    dim = tens.ndim
    vecs = map(lambda i: np.einsum(stringEin, np.swapaxes(tens, 0, i), *([vec] * (tensOrder - 1))), range(dim))
    vecNew = functools.reduce(lambda x, y: x + y, vecs) / dim
    vecNorm = np.linalg.norm(vecNew)
    return vecNew / vecNorm, vecNorm


def power_iteration_symmetrizing(self, tens, vecGuess, maxIter, tol):
    """ Function to perform higher order power iteration on a non-symmetric tensor

    Args:
        tens (np array)
            Tensor

        vec (np array)
            Vector

        maxIter (int)
            Max number of iterations to perform

        tol (float)
            Tolerance for difference and iterates

    Returns:
        eigVec (np array)

        eigValue (np array)
    """
    stringEin = self.power_iterate_string(tens)
    tensOrder = tens.ndim
    vec = None
    vecNorm = None
    for i in range(maxIter):
        vecPrev = vecGuess
        vec, vecNorm = self.power_iterate_symmetrizing(stringEin, tensOrder, tens, vecPrev)
        if np.linalg.norm(vec - vecPrev) < tol:
            break
    return vec, vecNorm


def nonlin_index_2_eigenvector(self, stm, stt):
    """ Function to calculate the nonlinearity index

    The maximum eigenvalue of the tensor squared

    Args:
        stm (np array)
            State transition matrix (used to generate guess)

        stt (np array)
            Arbitrary order state transition tensor

    Returns:
        nonlinearity_index (float)
    """
    _, _, vh = svd(stm)
    stmVVec = vh[0, :]
    tensSquared = np.einsum(self.tensor_square_string(stt), stt, stt)
    tensSquaredSym = self.symmetrize_tensor(tensSquared)
    _, sttNorm = self.power_iteration(tensSquaredSym, stmVVec, 20, 1e-3)
    stmNorm = norm(stm, 2)
    return math.sqrt(sttNorm) / stmNorm


def nonlin_index_2_eigenvector_symmetrizing(self, stm, stt):
    """ Function to calculate the nonlinearity index

    The maximum eigenvalue of the tensor squared computed with symmetrization along the way

    Args:
        stm (np array)
            State transition matrix (used to generate guess)

        stt (np array)
            Arbitrary order state transition tensor

    Returns:
        nonlinearity_index (float)
    """
    _, _, vh = svd(stm)
    stmVVec = vh[0, :]
    tensSquared = np.einsum(self.tensor_square_string(stt), stt, stt)
    # tensSquaredSym = self.symmetrize_tensor(tensSquared)
    _, sttNorm = self.power_iteration_symmetrizing(tensSquared, stmVVec, 20, 1e-3)
    stmNorm = norm(stm, 2)
    return math.sqrt(sttNorm) / stmNorm


def stt_2_norm(self, stm, stt):
    """ Function to calculate the norm of a state transition tensor

    The maximum eigenvalue of the tensor squared

    Args:
        stm (np array)
            State transition matrix (used to generate guess)

        stt (np array)
            Arbitrary order state transition tensor

    Returns:
        2-norm of the tensor (float)
    """
    # generate initial guess from maximum right singular vector of stm
    _, _, vh = svd(stm)
    stmVVec = vh[0, :]
    tensSquared = np.einsum(self.tensor_square_string(stt), stt, stt)
    _, sttNorm = self.power_iteration(tensSquared, stmVVec, 20, 1e-3)
    return math.sqrt(sttNorm)


def tensor_2_norm(self, tens, guessVec):
    """ Function to calculate the norm of a state transition tensor

    The square root of the maximum eigenvalue of the tensor squared

    Args:
        tens (np array)
            Arbitrary 1-m tensor
        guessVec (np array)
            Guess vector for input that maximizes the tensor

    Returns:
        nonlinearity_index (float)
    """
    tensSquared = np.einsum(self.tensor_square_string(tens), tens, tens)
    _, tensNorm = self.power_iteration(tensSquared, guessVec, 20, 1e-3)
    return math.sqrt(tensNorm)


def cocycle1(self, stm10, stm21):
    """ Function to find STM along two combined subintervals

   The cocycle conditon equation is used to find Phi(t2,t_0)=Phi(t2,t_1)*Phi(t1,t_0)

    Args:
        stm10 (np array)
            State transition matrix from time 0 to 1

        stm21 (np array)
            State transition matrix from time 1 to 2

    Returns:
        stm20 (np array)
            State transition matrix from time 0 to 2
    """
    stm20 = np.matmul(stm21, stm10)

    return stm20


def cocycle2(self, stm10, stt10, stm21, stt21):
    """ Function to find STM and STT along two combined subintervals

   The cocycle conditon equation is used to find Phi(t2,t0)=Phi(t2,t1)*Phi(t1,t0)
    and the generalized cocycle condition is used to find Psi(t2,t0)

    Args:
        stm10 (np array)
            State transition matrix from time 0 to 1

        stt10 (np array)
            State transition tensor from time 0  to 1

        stm21 (np array)
            State transition matrix from time 1 to 2

        stt21 (np array)
            State transition tensor from time 1 to 2

    Returns:
        stm20 (np array)
            State transition matrix from time 0 to 2

        stt20 (np array)
            State transition tensor from time 0 to 2
    """
    stm20 = np.matmul(stm21, stm10)
    stt20 = np.einsum('il,ljk->ijk', stm21, stt10) + np.einsum('ilm,lj,mk->ijk', stt21, stm10, stm10)

    return [stm20, stt20]



