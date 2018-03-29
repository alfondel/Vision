/*
	File: P3.cpp
		
	Authors:
		Alfonso Delgado Vellosillo - 679745
		Daniel Martinez Martinez - 538798
*/


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include <vector>


int main(int argc, char * argv[]) {

	std::string file;
	std::string program = argv[0];
	std::string err = "USE: "+ program +" f <filename>\n";
	if (argc < 1) {
		std::cout << err;
		
		exit(1);
	}
	file = argv[2];

	cv::Mat image = cv::imread(file, CV_LOAD_IMAGE_COLOR);

	imshow("Imagen", image);
	
	//Para sobel https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html

	//Finish program
	std::cout << "Pulsa una tecla para terminar ";
	cv::waitKey(0);
}