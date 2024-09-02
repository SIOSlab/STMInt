import numpy as np
import math
from scipy.linalg import eigh, norm, svd
import string
import itertools
import functools


# ======================================================================================================================
# Nonlinearity Index Functions
# ======================================================================================================================


def nonlin_index_inf_2(stm, stt):
    """Function to calculate the nonlinearity index

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


def nonlin_index_junkins_scale_free(stm, stt):
    """Function to calculate the nonlinearity index

    The induced 2 norm of the unfolded STT is used in this calculation
    This gives the quotient of the (Frobenius, 2)-norm of the second order STT
    with the Frobenius norm of the STM.

     Args:
         stm (np array)
             State transition matrix

         stt (np array)
             Second order state transition tensor

     Returns:
         nonlinearity_index (float)
    """
    dim = len(stm)
    sttNorm = norm(np.reshape(stt, (dim**2, dim)), 2)
    stmNorm = norm(stm, "fro")
    return sttNorm / stmNorm


def nonlin_index_unfold_bound(stm, stt):
    """Function to calculate the nonlinearity index

    The induced 2 norm of the unfolded STT is used in this calculation.
    This is a bound on the 2-norm of the second order STT quotiented with
    the 2-norm of the STM

     Args:
         stm (np array)
             State transition matrix

         stt (np array)
             Second order state transition tensor

     Returns:
         nonlinearity_index (float)
    """
    dim = len(stm)
    sttNorm = norm(np.reshape(stt, (dim, dim**2)), 2)
    stmNorm = norm(stm, 2)
    return sttNorm / stmNorm


# ======================================================================================================================
# Power Iteration Functions
# ======================================================================================================================


def power_iterate_string(tens):
    """Function to calculate the index string for einsum (up to 26 dimensional tensor)

    Args:
        tens (np array)
            Tensor

    Returns:
        einsum string to perform power iteration (string)
    """
    assert tens.ndim <= 26
    # looks like "zabcd,a,b,c,d->z"
    stringEin = "z"
    stringContract = string.ascii_lowercase[: tens.ndim - 1]
    secondString = ""
    for char in stringContract:
        secondString += "," + char
    stringEin += stringContract + secondString + "->" "z"
    return stringEin


def tensor_square_string(tens):
    """Function to calculate the index string for einsum (up to 1-13 dimensional tensor)
    Args:
        tens (np array)
            Tensor

    Returns:
        einsum string to perform tensor squaring (string)
    """
    assert tens.ndim < 13
    # looks like "abcd,azyx-bcdzyx>"
    firstString = string.ascii_lowercase[1 : tens.ndim]
    secondString = string.ascii_lowercase[26 : 26 - tens.ndim : -1]
    stringEin = (
        "a" + firstString + ",a" + secondString + "->" + firstString + secondString
    )
    return stringEin


def power_iterate(stringEin, tensOrder, tens, vec):
    """Function to perform one higher order power iteration on a symmetric tensor

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


def power_iteration(tens, vecGuess, maxIter, tol):
    """Function to perform higher order power iteration on a symmetric tensor

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
    stringEin = power_iterate_string(tens)
    tensOrder = tens.ndim
    vec = vecGuess
    vecNorm = None
    for i in range(maxIter):
        vecPrev = vec
        vec, vecNorm = power_iterate(stringEin, tensOrder, tens, vecPrev)
        if np.linalg.norm(vec - vecPrev) < tol:
            break
    return vec, vecNorm


def power_iterate_symmetrizing(stringEin, tensOrder, tens, vec):
    """Function to perform one higher order power iteration on a non-symmetric tensor

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
    vecs = map(
        lambda i: np.einsum(
            stringEin, np.swapaxes(tens, 0, i), *([vec] * (tensOrder - 1))
        ),
        range(dim),
    )
    vecNew = functools.reduce(lambda x, y: x + y, vecs) / dim
    vecNorm = np.linalg.norm(vecNew)
    return vecNew / vecNorm, vecNorm


def power_iteration_symmetrizing(tens, vecGuess, maxIter, tol):
    """Function to perform higher order power iteration on a non-symmetric tensor

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
    stringEin = power_iterate_string(tens)
    tensOrder = tens.ndim
    vec = vecGuess
    vecNorm = None
    for i in range(maxIter):
        vecPrev = vec
        vec, vecNorm = power_iterate_symmetrizing(stringEin, tensOrder, tens, vecPrev)
        if np.linalg.norm(vec - vecPrev) < tol:
            break
    return vec, vecNorm


def get_polynomial_bound(tens):
    """Function to find a bound on the value of a sclar valued polynomial on the unit sphere

    Args:
        tens (np array)
            Tensor

    Returns:
        K (double)
            Bound on the polynomial on the unit sphere
    """
    tensOrder = tens.ndim
    return np.sum(np.abs(tens)) * ((tensOrder - 1.0) * tensOrder)


def MM_iterate(stringEin, tensOrder, tens, K, vec):
    """Function to perform one step of polynomial optimization on a scalar valued polynomial on unit sphere

    Single step

    Args:
        stringEin (string)
            String to instruct einsum to perform contractions

        tensOrder (int)
            Order of the tensor

        tens (np array)
            Tensor

        K (double)
            Damping constant

        vec (np array)
            Vector

    Returns:
        vecNew (np array)

        vecNorm (float)
    """
    poly = np.einsum(stringEin, tens, *([vec] * (tensOrder - 1)))
    vecNew = vec - 1 / K * poly
    vecNorm = np.linalg.norm(vecNew)
    return vecNew / vecNorm, vecNorm


def MM_iteration(tens, vecGuess, maxIter, tol):
    """Function to perform polynomial optimization on a scalar valued polynomial on unit sphere

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
    stringEin = power_iterate_string(tens)
    tensOrder = tens.ndim
    vec = vecGuess
    vecNorm = None
    K = get_polynomial_bound(tens)
    for i in range(maxIter):
        vecPrev = vec
        vec, vecNorm = MM_iterate(stringEin, tensOrder, tens, K, vecPrev)
        if np.linalg.norm(vec - vecPrev) < tol:
            break
    return vec, vecNorm


def symmetrize_tensor(tens):
    """Symmetrize a tensor

    Args:
        tens (np array)
            Tensor

    Returns:
        symTens (np array)
    """
    dim = tens.ndim
    rangedim = range(dim)
    tensDiv = tens / math.factorial(dim)
    permutes = map(
        lambda sigma: np.moveaxis(tensDiv, rangedim, sigma),
        itertools.permutations(range(dim)),
    )
    symTens = functools.reduce(lambda x, y: x + y, permutes)
    return symTens


def nonlin_index_2(stm, stt):
    """Function to calculate the nonlinearity index

    Using tensor eigenvalues, the quotient of the induced 2-norm
    of the STT with the 2-norm of the STM

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
    tensSquared = np.einsum(tensor_square_string(stt), stt, stt)
    _, sttNorm = power_iteration(tensSquared, stmVVec, 20, 1e-3)
    stmNorm = norm(stm, 2)
    return math.sqrt(sttNorm) / stmNorm


def nonlin_index_DEMoN2(stm, stt):
    """Function to calculate the nonlinearity index

    Using tensor eigenvalues, the quotient of the induced 2-norm
    of the STT with the 2-norm of the STM

    Args:
        stm (np array)
            State transition matrix (used to generate guess)

        stt (np array)
            Arbitrary order state transition tensor

    Returns:
        nonlinearity_index (float)
    """
    maxdemon = 0
    istm = np.linalg.inv(stm)
    tens = np.einsum("ilm,lj,mk->ijk", stt, istm, istm)
    tensSquared = np.einsum("ijk,ilm->jklm", tens, tens)
    for i in range(100):
        guess = np.random.multivariate_normal(np.zeros(len(stm)), np.identity(len(stm)))
        guess = guess / np.linalg.norm(guess)
        argMax, m_1norm = power_iteration(tensSquared, guess, 300, 1e-9)
        argMax = np.matmul(istm, argMax)
        argMax = argMax / np.linalg.norm(argMax)
        demon = np.linalg.norm(
            np.einsum("ijk,j,k->i", stt, argMax, argMax)
        ) / np.linalg.norm(np.einsum("ij,j->i", stm, argMax))
        maxdemon = max(demon, maxdemon)
    return maxdemon


def nonlin_index_TEMoN3(stm, stt):
    """Function to calculate the nonlinearity index

    Using tensor eigenvalues, the quotient of the induced 2-norm
    of the 3rd order term in the CGT series with the 2-norm of the second order term in the CGT series

    Args:
        stm (np array)
            State transition matrix (used to generate guess)

        stt (np array)
            Arbitrary order state transition tensor

    Returns:
        nonlinearity_index (float)
    """
    maxtemon = 0
    istm = np.linalg.inv(stm)
    CGT3 = np.einsum("lij,lk->ijk", stt, stm)
    tens = np.einsum("lmn,li,mj,nk->ijk", CGT3, istm, istm, istm)
    tens = symmetrize_tensor(tens)
    K = get_polynomial_bound(tens)
    for i in range(100):
        guess = np.random.multivariate_normal(np.zeros(len(stm)), np.identity(len(stm)))
        guess = guess / np.linalg.norm(guess)
        argMax, m_1norm = MM_iteration(-1.0 * tens, guess, 800, 1e-9)
        argMax = np.matmul(istm, argMax)
        argMax = argMax / np.linalg.norm(argMax)
        temon = (
            np.abs(np.einsum("ijk,i,j,k->", CGT3, argMax, argMax, argMax))
            / np.linalg.norm(np.einsum("ij,j->i", stm, argMax)) ** 2
        )

        argMax1, m_1norm = MM_iteration(tens, guess, 800, 1e-9)
        argMax1 = np.matmul(istm, argMax1)
        argMax1 = argMax1 / np.linalg.norm(argMax1)
        temon1 = (
            np.abs(np.einsum("ijk,i,j,k->", CGT3, argMax1, argMax1, argMax1))
            / np.linalg.norm(np.einsum("ij,j->i", stm, argMax1)) ** 2
        )
        maxtemon = max(temon, maxtemon)
        maxtemon = max(temon1, maxtemon)
    return maxtemon


def stt_2_norm(stm, stt):
    """Function to calculate the norm of the state transition tensor, and the input unit vector that leads to that norm.

    The maximum eigenvalue of the tensor squared computed with symmetrization along the way

    Args:
        stm (np array)
            State transition matrix

        stt (np array)
            Second order state transition tensor

    Returns:
        sttArgMax (np array)
            Input unit vector that maximizes the STT
        sqrt(sttNorm) (float)
            Square root of the norm of the STT
    """
    _, _, vh = svd(stm)
    stmVVec = vh[0, :]
    tensSquared = np.einsum("ijk,ilm->jklm", stt, stt)
    sttArgMax, sttNorm = power_iteration(tensSquared, stmVVec, 20, 1e-3)
    return sttArgMax, np.sqrt(sttNorm)


def tensor_2_norm(tens, guessVec):
    """Function to calculate the norm of a state transition tensor

    The square root of the maximum eigenvalue of the tensor squared

    Args:
        tens (np array)
            Arbitrary 1-m tensor
        guessVec (np array)
            Guess vector for input that maximizes the tensor

    Returns:
        nonlinearity_index (float)
    """
    tensSquared = np.einsum(tensor_square_string(tens), tens, tens)
    _, tensNorm = power_iteration(tensSquared, guessVec, 20, 1e-3)
    return math.sqrt(tensNorm)


def cocycle1(stm10, stm21):
    """Function to find STM along two combined subintervals

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


def cocycle2(stm10, stt10, stm21, stt21):
    """Function to find STM and STT along two combined subintervals

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
    stt20 = np.einsum("il,ljk->ijk", stm21, stt10) + np.einsum(
        "ilm,lj,mk->ijk", stt21, stm10, stm10
    )

    return [stm20, stt20]
