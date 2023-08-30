#include <iostream>
// #include <opencv2/opencv.hpp>
// #include <opencv4/opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

// error adding symbols: DSO missing from command line
 // undefined reference to symbol '_ZN2cv6imreadERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEi'

int main() {
    // Read Image
    cv::Mat img = cv::imread("noaa2.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
	std::cerr << "Error loading image" << std::endl;
	return -1;
    }

    // Apply Morphological Operations: Dilatation and Erosion
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    cv::Mat denoisedImage;
    cv::morphologyEx(img, denoisedImage, cv::MORPH_OPEN, element);

    // Display and Save Results
    cv::namedWindow("Original Image", cv::WINDOW_NORMAL);
    cv::imshow("Original Image", img);

    cv::namedWindow("Denoised Image", cv::WINDOW_NORMAL);
    cv::imshow("Denoised Image", denoisedImage);
    cv::imwrite("denoised_image.jpg", denoisedImage);

    return 1;
}

