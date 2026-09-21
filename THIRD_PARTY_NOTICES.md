# Third-party assets

The handwritten digit benchmark is the Optical Recognition of Handwritten Digits dataset by E. Alpaydin and C. Kaynak (1998), distributed through UCI under CC BY 4.0: https://doi.org/10.24432/C50P49. Study C uses all 1,797 images in the subset bundled with scikit-learn. It derives standardized features, a three-dimensional PCA representation, spherical coordinates, and descriptive query-weighted fits. The analysis does not modify the original dataset attribution.

The executed code depends on NumPy, SciPy, scikit-learn, Matplotlib, and threadpoolctl. NumPy, SciPy, scikit-learn, and threadpoolctl use BSD-style licenses. Matplotlib uses its own PSF-based license agreement. Joblib is an indirect scikit-learn dependency with a BSD license. Copies of the installed distributions' top-level license files appear in `licenses/`; these files also retain bundled-component notices where provided. Python itself uses the PSF license and is not redistributed here. Package binaries are not included.

The research code is distributed under the MIT license in `LICENSE`. The manuscript and conference style files are not distributed in this repository. Notebook tools and pandas are used for execution and presentation; their package distributions retain their own licenses.
