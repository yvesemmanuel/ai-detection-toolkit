"""
Image Filtering Module

This module provides various image filtering functions for edge detection and feature extraction.
Each filter applies different convolution kernels to highlight specific image features.
The module includes individual filter functions and a combined function to apply all filters.

Functions:
    apply_filter_a: Applies a set of 8 directional edge detection filters
    apply_filter_b: Applies a set of 8 complex edge detection filters
    apply_filter_c: Applies a set of 4 Laplacian-like filters
    apply_filter_d: Applies a set of 4 second-order derivative filters
    apply_filter_e: Applies a set of 4 high-frequency detail enhancement filters
    apply_filter_f: Applies a single Laplacian filter
    apply_filter_g: Applies a single high-frequency detail enhancement filter
    apply_all_filters: Combines all filters and applies thresholding
"""

import numpy as np
import cv2


def apply_filter_a(
    src: np.ndarray, normalize: bool = True, divisor: int = 8
) -> np.ndarray:
    """
    Applies a set of 8 directional edge detection filters.

    Parameters:
    -----------
    src: Input image as numpy array
    normalize: Whether to normalize the output
    divisor: Divisor for normalization

    Returns:
    --------
    Filtered image with edges enhanced in 8 directions.
    """
    src_copy = np.copy(src)
    f1 = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, -1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ]
    )

    img = cv2.filter2D(src=src_copy, kernel=f1[0], ddepth=-1)
    for filter in f1[1:]:
        img = cv2.add(img, cv2.filter2D(src=src_copy, kernel=filter, ddepth=-1))

    return img // divisor if normalize else img


def apply_filter_b(
    src: np.ndarray, normalize: bool = True, divisor: int = 8
) -> np.ndarray:
    """
    Applies a set of 8 complex edge detection filters.

    Parameters:
    -----------
    src: Input image as numpy array
    normalize: Whether to normalize the output
    divisor: Divisor for normalization

    Returns:
    --------
    Filtered image with complex edges enhanced.
    """
    src_copy = np.copy(src)
    f2 = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 2, 1, 0, 0],
                [0, 1, -3, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, -1, 0, 0],
                [0, 0, 3, 0, 0],
                [0, 0, -3, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 2, 0],
                [0, 0, -3, 1, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -3, 3, -1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, -3, 1, 0],
                [0, 0, 1, 2, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, -3, 0, 0],
                [0, 0, 3, 0, 0],
                [0, 0, -1, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 1, -3, 0, 0],
                [0, 2, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [-1, 3, -3, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ]
    )

    img = cv2.filter2D(src=src_copy, kernel=f2[0], ddepth=-1)
    for filter in f2[1:]:
        img = cv2.add(img, cv2.filter2D(src=src_copy, kernel=filter, ddepth=-1))

    return img // divisor if normalize else img


def apply_filter_c(
    src: np.ndarray, normalize: bool = True, divisor: int = 4
) -> np.ndarray:
    """
    Applies a set of 4 Laplacian-like filters for edge detection.

    Parameters:
    -----------
    src: Input image as numpy array
    normalize: Whether to normalize the output
    divisor: Divisor for normalization

    Returns:
    --------
    Filtered image with edges enhanced using Laplacian-like filters.
    """
    src_copy = np.copy(src)
    f3 = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, -2, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, -2, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, -2, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ]
    )

    img = cv2.filter2D(src=src_copy, kernel=f3[0], ddepth=-1)
    for filter in f3[1:]:
        img = cv2.add(img, cv2.filter2D(src=src_copy, kernel=filter, ddepth=-1))

    return img // divisor if normalize else img


def apply_filter_d(
    src: np.ndarray, normalize: bool = True, divisor: int = 4
) -> np.ndarray:
    """
    Applies a set of 4 second-order derivative filters.

    Parameters:
    -----------
    src: Input image as numpy array
    normalize: Whether to normalize the output
    divisor: Divisor for normalization

    Returns:
    --------
    Filtered image with second-order derivatives enhanced.
    """
    src_copy = np.copy(src)
    f4 = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 2, -4, 2, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, -1, 2, 0, 0],
                [0, 2, -4, 0, 0],
                [0, -1, 2, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 2, -1, 0],
                [0, 0, -4, 2, 0],
                [0, 0, 2, -1, 0],
                [0, 0, 0, 0, 0],
            ],
        ]
    )

    img = cv2.filter2D(src=src_copy, kernel=f4[0], ddepth=-1)
    for filter in f4[1:]:
        img = cv2.add(img, cv2.filter2D(src=src_copy, kernel=filter, ddepth=-1))

    return img // divisor if normalize else img


def apply_filter_e(
    src: np.ndarray, normalize: bool = True, divisor: int = 4
) -> np.ndarray:
    """
    Applies a set of 4 high-frequency detail enhancement filters.

    Parameters:
    -----------
    src: Input image as numpy array
    normalize: Whether to normalize the output
    divisor: Divisor for normalization

    Returns:
    --------
    Filtered image with high-frequency details enhanced.
    """
    src_copy = np.copy(src)
    f5 = np.array(
        [
            [
                [1, 2, -2, 2, 1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [1, 2, -2, 0, 0],
                [2, -6, 8, 0, 0],
                [-2, 8, -12, 0, 0],
                [2, -6, 8, 0, 0],
                [1, 2, -2, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [1, 2, -2, 2, 1],
            ],
            [
                [0, 0, -2, 2, 1],
                [0, 0, 8, -6, 2],
                [0, 0, -12, 8, -2],
                [0, 0, 8, -6, 2],
                [0, 0, -2, 2, 1],
            ],
        ]
    )

    img = cv2.filter2D(src=src_copy, kernel=f5[0], ddepth=-1)
    for filter in f5[1:]:
        img = cv2.add(img, cv2.filter2D(src=src_copy, kernel=filter, ddepth=-1))

    return img // divisor if normalize else img


def apply_filter_f(src: np.ndarray, normalize: bool = False) -> np.ndarray:
    """
    Applies a single Laplacian filter for edge detection.

    Parameters:
    -----------
    src: Input image as numpy array
    normalize: Whether to normalize the output (not used in this function but kept for API consistency)

    Returns:
    --------
    Filtered image with edges enhanced using a Laplacian filter.
    """
    src_copy = np.copy(src)
    f5 = np.asarray(
        [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    img = cv2.filter2D(src=src_copy, kernel=f5, ddepth=-1)
    return img


def apply_filter_g(src: np.ndarray, normalize: bool = False) -> np.ndarray:
    """
    Applies a single high-frequency detail enhancement filter.

    Parameters:
    -----------
    src: Input image as numpy array
    normalize: Whether to normalize the output (not used in this function but kept for API consistency)

    Returns:
    --------
    Filtered image with high-frequency details enhanced.
    """
    src_copy = np.copy(src)
    f5 = np.asarray(
        [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ]
    )

    img = cv2.filter2D(src=src_copy, kernel=f5, ddepth=-1)
    return img


def apply_all_filters(
    src: np.ndarray,
    threshold_offset: float = 2.0,
    normalize: bool = True,
    divisor: int = 7,
) -> np.ndarray:
    """
    Combines all filters and applies thresholding to create a binary image.

    Parameters:
    -----------
    src: Input image as numpy array
    threshold_offset: Offset added to the median for thresholding
    normalize: Whether to normalize the combined output
    divisor: Divisor for normalization

    Returns:
    --------
    Binary image after applying all filters and thresholding.
    """
    src_copy = np.copy(src)
    img = np.array(
        cv2.cvtColor(
            (
                apply_filter_a(src_copy)
                + apply_filter_b(src_copy)
                + apply_filter_c(src_copy)
                + apply_filter_d(src_copy)
                + apply_filter_e(src_copy)
                + apply_filter_f(src_copy)
                + apply_filter_g(src_copy)
            ),
            cv2.COLOR_RGB2GRAY,
        )
        // divisor
        if normalize
        else 1
    )
    img_thresh = np.median(img) + threshold_offset
    return cv2.threshold(img, img_thresh, 255, cv2.THRESH_BINARY)[1]
