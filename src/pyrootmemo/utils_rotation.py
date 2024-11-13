## FUNCTIONS FOR 3-DIMENSIONAL ROTATIONS, USING AXIS-ANGLE VECTORS
## 11/11/2024 - GJM

# load packages
import numpy as np


# normalise a vector, or array of vectors
def vector_normalise(
        vector: np.ndarray
        ) -> np.ndarray:
    """
    Normalise a (array of) vectors. In other words, scale them so their length
    is equal to 1. 

    Parameters
    ----------
    vector : np.ndarray
        Array with vectors. Vectors are normalised over their last axis.

    Returns
    -------
    Array with normalised vectors (np.ndarray, same size as input 'vector')

    """
    
    # vector lengths
    vector_length = np.linalg.norm(vector, axis = -1)
    # return normalised values
    return(vector / np.reshape(vector_length, (*vector.shape[:-1], 1)))

    
# Split axis-angle rotation vector into a seperate unit vector and rotation magnitude    
def axisangle_split(
        axisangle: np.ndarray
        ) -> tuple:
    """
    Split an 3-dimensional axis-angle rotation vector into its unit vector 
    describing the axis of rotation, and the rotation magnitude

    Parameters
    ----------
    axisangle : np.ndarray(3)
        3-dimensional axis-angle rotation vector.

    Returns
    -------
    tuple
        Tuple containing a unit vector (np.ndarray, size 3) and a scalar
        describing the rotation magnitude

    """
    # check input
    _axisangle_check()
    # calculate rotation magnitude
    rotation_magnitude = np.linalg.norm(axisangle)
    # get unit vector describing the axis of rotation
    if np.isclose(rotation_magnitude, 0.):
        rotation_axis = np.zeros(3)
    else:
        rotation_axis = axisangle/rotation_magnitude
    # return axis and magnitude
    return(rotation_axis, rotation_magnitude)
    
    
# rotate (an array of) vector using a axis-angle rotation vector and Rodrigues' formula
def axisangle_rotate(
        vector: np.ndarray,
        axisangle: np.ndarray
        ) -> np.ndarray:
    """
    Rotate a (series of) three-dimensional vectors, using a rotation described
    by an axis-angle vector, using Rodrigues' formula (follows definition
    on see https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula)

    Parameters
    ----------
    vector : np.ndarray (size 3, or size n*3)
        3-dimensional vector, or array of three dimensional vectors.
    axisangle : np.ndarray (size 3)
        axis-angle vector describing the three-dimensional rotation using a
        rotation axis and rotation magnitude (length of vector)

    Returns
    -------
    np.ndarray with rotated vectors (same size as input 'vector').

    """
    # check vector
    _vector_check()
    # get axis-angle rotation vector magnitude (theta) and unit vector (n)
    rotation_axis, rotation_magnitude = axisangle_split(axisangle)
    # apply rodrigues formula
    vector_rotated = (
        vector * np.cos(rotation_magnitude) 
        + np.cross(vector, rotation_axis) * np.sin(rotation_magnitude) 
        + np.outer(np.dot(vector, rotation_axis), rotation_axis)
        * (1. - np.cos(rotation_magnitude))
        )
    # return array - remove empty dimensions
    return(vector_rotated.squeeze())
        
    
# convert axis-angle vector to a rotation matrix R)
def axisangle_to_rotationmatrix(
        axisangle: np.ndarray
        ) -> np.ndarray:
    """
    Convert rotation described by axis-angle vector to a three-dimensional
    rotation matrix. Follows definitions as described on
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    Parameters
    ----------
    axisangle : np.ndarray (size 3)
        Axis-angle vector describing three-dimensional rotation.

    Returns
    -------
    np.ndarray (size 3*3) with the corresponding rotation matrix.

    """
    # check input
    _axisangle_check()
    # get axis-angle rotation vector magnitude and unit vector
    rotation_magnitude = np.linalg.norm(axisangle)
    if np.isclose(rotation_magnitude, 0.):
        rotation_axis = np.zeros(3)
    else:
        rotation_axis = axisangle/rotation_magnitude
    # matrix K
    K = np.array([
        [0., -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0., -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0.],
        ])
    # rotation matrix
    R = (np.eye(3) 
         + np.sin(rotation_magnitude)*K 
         + (1. - np.cos(rotation_magnitude)) * K @ K)
    # return
    return(R)
        
    
# convert a rotation matrix to an axis-angle rotation vector
def rotationmatrix_to_axisangle(
        matrix: np.ndarray
        ) -> np.ndarray:
    """
    Convert three-dimensional rotation matrix to axis-angle vector
    representation. Follows theory as described in
    https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation

    Parameters
    ----------
    matrix : np.ndarray (size 3*3)
        three-dimensional rotation matrix.

    Returns
    -------
    axis-angle vector (np.array, size 3)

    """
    # input check
    _rotationmatrix_check()    
    # rotation magnitude
    rotation_magnitude = np.arccos((np.trace(matrix) - 1.) / 2.)
    # solving method depends on whether the rotation matrix is symmetric
    if np.allclose(matrix, matrix.T):
        # symmetric case - https://en.wikipedia.org/wiki/Rotation_matrix#Determining_the_axis
        Warning('symmetric rotation matrix - function not yet implemented')
    else:
        # non-symmetric case
        # rotation axis unit vector
        rotation_axis = np.array([
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1]
            ]) / (2. * np.sin(rotation_magnitude))
        # return axis-angle representation
        return(rotation_axis * rotation_magnitude)


# check axisangle vector - must be numpy array size 3
def _axisangle_check(
        axisangle = np.ndarray
        ) -> None:
    if not isinstance(axisangle, np.ndarray):
        raise TypeError('axis-angle vector must be defined as a numpy array')
    if axisangle.shape != (3, ):
        raise ValueError('axis-angle vector must have length 3')
        
        
# check (array of) 3-D vector - must be numpy array size 3 or n*3        
def _vector_check(
        vector = np.ndarray
        ) -> None:
    if not isinstance(vector, np.ndarray):
        raise TypeError('vector must be defined as a numpy array')
    if not (vector.shape == (3, ) or ((vector.ndim == 2) and (vector.shape[1] == 3))):
        raise ValueError('vector must have length 3 (single vector), or n*3 (n vectors)')


# check 3-D rotation matrix - must be numpy array size 3*3
def _rotationmatrix_check(
        matrix: np.ndarray
        ) -> None:
    if not isinstance(matrix, np.ndarray):
        raise TypeError('matrix must be defined as a numpy array')
    if not matrix.shape == (3, 3):
        raise ValueError('matrix must have shape 3*3')
