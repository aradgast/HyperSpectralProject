# HyperSpectralProject
Artificial hyperspectral datasets for hyperspectral target detection

Abstract

Hyperspectral imagery are images taken at many spectral bands. Our focus is getting data from the behavior of the pixel vector to detect targets.

A suggested procedure for checking new detection algorithms in the field is by using them on artificial image.

Our method of creating the artificial image is performing Principal Component Analysis (PCA) on the original image so it would have the same statistic parameters as the original image. Then, we create the image based on the expected value and the covariance of the original image, using T-distribution functions.

Our main goal in the project is to verify that the artificial image we create is valid, meaning that we can test algorithms on it and expect that the performance of the known algorithms will be similar as the on in the original image.

If we will manage to accomplish our main goal, we will expand the research by using more target detection algorithms and more images.

Keywords: Hyperspectral image, Target detection, Artificial image, PCA, T-distribution, Statistical analysis, Detection algorithms

# How to use this repository and create your own artificial image??

## 1. Clone the repository
## 2. Download the image you want to create an artificial image from and add to your local repository
## 3. Create an instance of the class "ArtificialImage" and pass the image path to the constructor
## 4. The __init__ function will create the artificial image. You can save the output by using save_cubes method.
## 5. You can use the MF implementation of the target detection algorithm by using the function MF in the matched_filter.py file.
## 6. You can plot the results by using the calc_stats and plot_results functions in the plot_detection_algo.py file.
## note: you can use the main_py.py file as a refernce