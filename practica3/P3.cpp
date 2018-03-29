/*
	File: P3.cpp
		
	Authors:
		Alfonso Delgado Vellosillo - 679745
		Daniel Martinez Martinez - 538798
*/

#define _USE_MATH_DEFINES
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;


void filtroGauss(Mat src, float t) {
	MatIterator_<Vec3b> it, end;
	int MAX_KERNEL_LENGTH = 2;
	Mat dst;
	for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
	{
		GaussianBlur(src, dst, Size(i, i), 0, 0);
	}
	imshow("Imagen FILTRO", dst);
}



int main(int argc, char * argv[]) {

	string file;
	string program = argv[0];
	string err = "USE: "+ program +" f <filename>\n";
	if (argc < 1) {
		std::cout << err;
		
		exit(1);
	}
	file = argv[2];

	Mat image = imread(file, CV_LOAD_IMAGE_COLOR);

	imshow("Imagen", image);
	
	//Para sobel https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html

	//Finish program
	std::cout << "Pulsa una tecla para terminar ";
	cv::waitKey(0);
}