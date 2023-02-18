# vision4
Frequency Domain
4.1. Fourier transform
4.1.1. For each filter given below, compute its Fourier transform, and illustrate its magnitude response. Determine what
is its function (smoothing, edge enhancement or edge detection) based on the filter coefficients as well as its
frequency response. For each filter, determine whether it is separable? If yes, compute the FT separately and
explain the function of each 1D filter. If not, compute the FT directly. (Test on grayscale Lena Image).
�)
1
16 [1 2 1 2 4 2 1 2 1] �) [− − −1 1 1 − −81 1 − − −1 1 1] �) [−0 01 − −51 1 −0 01]
4.1.2. Perform 2D DFT on grayscale Lena, Barbara, F16, and Baboon images. Display the magnitude of the DFT image
with and without shifting and with and without logarithmic. Display and discuss the results. Also, examine in
which frequency range the DFT coefficients have large magnitudes and explain why?
4.2. Filtering
4.2.1. *Use DFT function to compute the linear convolution of an image �(�, �) with a filter �(�, �). Let the convolved
image be denoted by �(�, �). Firstly, suppose the image size is 256 × 256 and the filter size is 11 × 11; What is
the required size of the DFT to obtain the convolution of these two? Explain the exact steps to obtain the
convolution result. Secondly, suppose we use a 256 × 256 point DFT algorithm for �(�, �) and �(�, �), and obtain
�(�, �) as � = ���� (���(�).∗ ���(�)). The DFT and IDFT in this equation are both 256 × 256 points. For what
values of (�, �) does �(�, �) equal �(�, �)?
4.2.2. Write a program that filters grayscale Barbara image by zeroing out certain DFT coefficients.
The program consists of three steps:
1. Performing 2D DFT.
2. Zeroing out the coefficients at certain frequencies (see below).
3. Performing inverse DFT to get back a filtered image.
Note: Truncate or scale the image properly such that its range is between 0 and 255.
For part 2, try the following two types of filters:
a. Let �(�, �) = 0 for �� < {�, �} < (1 − � )�, � = 1/4, 1/8 (low-pass filtering).
b. Let �(�, �) = 0 for the following regions:i. 0 ≤ {� ��� �} ≤ ��;
ii. 0 <= � <= ��, ��� (1 − �)� ≤ � ≤ � − 1;
iii. (1 − �)� ≤ � ≤ � − 1 ��� 0 ≤ {�} ≤ ��;
iv. (1 − �)� ≤ � ��� � ≤ � − 1; � = 1/4, 1/8
Display and compare the original and processed images. Discuss the function of the two types of filters.
Note: you can use fft2, ifft2, fftshift, and rgb2gray functions for problem 4.
