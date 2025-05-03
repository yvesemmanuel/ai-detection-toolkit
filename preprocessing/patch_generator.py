"""
Patch Generator Module

This module provides functionality to split images into patches, analyze texture richness,
and reconstruct images based on texture characteristics. It works in conjunction with the
filters module for advanced image analysis.

The implementation is based on the paper: https://arxiv.org/abs/2311.12397
"""

import tensorflow as tf
import PIL.Image
import cv2
import numpy as np
import random
import concurrent.futures
from typing import List, Tuple
import preprocessing.filters as f


class PatchGenerator:
    """Handles image patch extraction, analysis and reconstruction."""

    def __init__(self, patch_size: int = 32, target_size: Tuple[int, int] = (256, 256)):
        """
        Initialize the patch generator.

        Parameters:
        -----------
        patch_size: Size of the patches to extract
        target_size: Target size for resizing input images
        """
        self.patch_size = patch_size
        self.target_size = target_size
        self.grid_dim = target_size[0] // patch_size

    def load_image(self, input_path: str) -> PIL.Image.Image:
        """
        Load and preprocess an image.

        Parameters:
        -----------
        input_path: Path to the input image

        Returns:
        --------
        Preprocessed PIL Image
        """
        if isinstance(input_path, bytes):
            input_path = input_path.decode("utf-8")
        img = PIL.Image.open(fp=input_path)
        if not (
            input_path.lower().endswith("jpg") or input_path.lower().endswith("jpeg")
        ):
            img = img.convert("RGB")

        if img.size != self.target_size:
            img = img.resize(size=self.target_size)

        return img

    def extract_patches(
        self, img: PIL.Image.Image
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extract patches from an image in both grayscale and color.

        Parameters:
        -----------
        img: Input PIL Image

        Returns:
        --------
        Tuple of (grayscale_patches, color_patches)
        """
        grayscale_patches = []
        color_patches = []

        for i in range(0, img.height, self.patch_size):
            for j in range(0, img.width, self.patch_size):
                box = (j, i, j + self.patch_size, i + self.patch_size)
                img_patch = img.crop(box)
                img_array = np.asarray(img_patch)

                if len(img_array.shape) == 2:
                    img_color = tf.image.grayscale_to_rgb(img_array)
                    grayscale_image = img_array.copy()
                elif img_array.shape[-1] == 3:
                    img_color = img_array.copy()
                    grayscale_image = cv2.cvtColor(src=img_color, code=cv2.COLOR_RGB2GRAY)
                else:
                    raise ValueError(f"Unsupported number of channels: {img_array.shape[-1]}")

                grayscale_patches.append(grayscale_image.astype(dtype=np.int32))
                color_patches.append(img_color)

        return grayscale_patches, color_patches

    def _calculate_pixel_variation(self, patch: np.ndarray) -> float:
        """
        Calculate pixel variation for a patch using horizontal, vertical,
        and diagonal differences.

        Parameters:
        -----------
        patch: Input patch as numpy array

        Returns:
        --------
        Pixel variation score
        """
        x, y = patch.shape

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_horizontal = executor.submit(self._horizontal_variation, patch, x, y)
            future_vertical = executor.submit(self._vertical_variation, patch, x, y)
            future_diagonal = executor.submit(self._diagonal_variation, patch, x, y)

            horizontal_var = future_horizontal.result()
            vertical_var = future_vertical.result()
            diagonal_var = future_diagonal.result()

        return horizontal_var + vertical_var + diagonal_var

    def _horizontal_variation(self, patch: np.ndarray, x: int, y: int) -> float:
        """Calculate horizontal pixel variation."""
        variation = 0
        for i in range(0, y - 1):
            for j in range(0, x):
                variation += abs(int(patch[j, i]) - int(patch[j, i + 1]))
        return variation

    def _vertical_variation(self, patch: np.ndarray, x: int, y: int) -> float:
        """Calculate vertical pixel variation."""
        variation = 0
        for i in range(0, y):
            for j in range(0, x - 1):
                variation += abs(int(patch[j, i]) - int(patch[j + 1, i]))
        return variation

    def _diagonal_variation(self, patch: np.ndarray, x: int, y: int) -> float:
        """Calculate diagonal pixel variation."""
        variation = 0
        for i in range(0, y - 1):
            for j in range(0, x - 1):
                variation += abs(int(patch[j, i]) - int(patch[j + 1, i + 1]))
                variation += abs(int(patch[j + 1, i]) - int(patch[j, i + 1]))
        return variation

    def classify_patches(
        self, patches: List[np.ndarray], texture_scores: List[float]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Classify patches as rich or poor texture based on texture scores.

        Parameters:
        -----------
        patches: List of image patches
        texture_scores: List of texture richness scores for each patch

        Returns:
        --------
        Tuple of (rich_texture_patches, poor_texture_patches)
        """
        threshold = np.mean(texture_scores)
        rich_texture_patches = []
        poor_texture_patches = []

        for i, score in enumerate(texture_scores):
            if score >= threshold:
                rich_texture_patches.append(patches[i])
            else:
                poor_texture_patches.append(patches[i])

        return rich_texture_patches, poor_texture_patches

    def reconstruct_image(
        self, patches: List[np.ndarray], is_color: bool = True
    ) -> np.ndarray:
        """
        Reconstruct a full image from patches.

        Parameters:
        -----------
        patches: List of patches to use in reconstruction
        is_color: Whether patches are in color

        Returns:
        --------
        Reconstructed image as numpy array
        """
        patches_copy = patches.copy()
        random.shuffle(patches_copy)

        required_patches = self.grid_dim * self.grid_dim
        while len(patches_copy) < required_patches:
            patches_copy.append(patches_copy[random.randint(0, len(patches) - 1)])

        if is_color:
            grid = np.array(patches_copy[:required_patches]).reshape(
                (self.grid_dim, self.grid_dim, self.patch_size, self.patch_size, 3)
            )
        else:
            grid = np.array(patches_copy[:required_patches]).reshape(
                (self.grid_dim, self.grid_dim, self.patch_size, self.patch_size)
            )

        rows = [np.concatenate(grid[i, :], axis=1) for i in range(self.grid_dim)]
        return np.concatenate(rows, axis=0)

    def process_image(
        self, input_path: str, use_filters: bool = False, is_color: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process an image to extract rich and poor texture reconstructions.

        Parameters:
        -----------
        input_path: Path to the input image
        use_filters: Whether to use filters module for texture analysis
        is_color: Whether to process in color

        Returns:
        --------
        Tuple of (rich_texture_image, poor_texture_image)
        """
        img = self.load_image(input_path)
        grayscale_patches, color_patches = self.extract_patches(img)

        texture_scores = []
        for patch in grayscale_patches:
            texture_scores.append(self._calculate_pixel_variation(patch))

        patches_to_use = color_patches if is_color else grayscale_patches
        rich_patches, poor_patches = self.classify_patches(
            patches_to_use, texture_scores
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            rich_future = executor.submit(
                self.reconstruct_image, rich_patches, is_color
            )
            poor_future = executor.submit(
                self.reconstruct_image, poor_patches, is_color
            )

            rich_texture = rich_future.result()
            poor_texture = poor_future.result()

        return rich_texture, poor_texture


def smash_n_reconstruct(
    input_path: str,
    use_filters: bool = False,
    is_color: bool = True,
    patch_size: int = 32,
    target_size: Tuple[int, int] = (256, 256),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Smash and Reconstruct preprocessing on an image.
    Reference: https://arxiv.org/abs/2311.12397

    Parameters:
    -----------
    input_path: Path to the input image
    patch_size: Size of the patches to extract
    use_filters: Whether to use filters module for texture analysis
    is_color: Whether to process in color

    Returns:
    --------
    Tuple of (rich_texture_image, poor_texture_image)
    """
    generator = PatchGenerator(patch_size=patch_size, target_size=target_size)
    return generator.process_image(input_path, use_filters, is_color)


def preprocess(path: str, label: int = None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Preprocess an image to extract rich and poor texture reconstructions.

    Parameters:
    -----------
    path: Path to the input image
    label: Label of the image

    Returns:
    --------
    Tuple of (rich_texture_image, poor_texture_image, label)
    """
    if isinstance(path, bytes):
        path = path.decode("utf-8")
    elif isinstance(path, tf.Tensor):
        path = path.numpy().decode("utf-8")

    rt, pt = smash_n_reconstruct(path)
    frt = tf.cast(tf.expand_dims(f.apply_all_filters(rt), axis=-1), dtype=tf.float64)
    fpt = tf.cast(tf.expand_dims(f.apply_all_filters(pt), axis=-1), dtype=tf.float64)
    if label is not None:
        label = tf.constant(label, dtype=tf.int32)
        return frt, fpt, label
    else:
        return frt, fpt
