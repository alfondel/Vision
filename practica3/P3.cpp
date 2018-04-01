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
#include <string.h>
#include <stdio.h>
using namespace cv;
using namespace std;

#define PI 3.14159265

void filtroGauss(Mat src, float t) {
	MatIterator_<Vec3b> it, end;
	int MAX_KERNEL_LENGTH = 20;
	Mat dst;
	for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
	{
		// ver https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#gaussianblur
		GaussianBlur(src, dst, Size(i, i), 0, 0);
	}
	imshow("Imagen FILTRO", dst);
}



int main(int argc, char * argv[]) {

	string file = "";
	string program = argv[0];
	string err = "USE: " + program + " -f <filename>\n";
	if (argc < 3) {
		std::cout << err;
		exit(1);
	}
	for (int i = 0; i < argc; i++) {
		if (strcmp(argv[i], "-f") == 0) {
			file = argv[i + 1];
		}

	}


	//Inicio del programa
	Mat image = imread(file, CV_LOAD_IMAGE_COLOR);

	imshow("Imagen", image);

	filtroGauss(image, 1);



	//Para sobel https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html
	Mat src, src_gray,src_gau;
	Mat grad;
	string window_name = "Sobel Demo - Simple Edge Detector";
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	int c;

	/// Load an image
	src = imread(file);

	if (!src.data)
	{
		return -1;
	}

	GaussianBlur(src, src_gau, Size(3, 3), 0, 0, BORDER_DEFAULT);

	/// Convert it to gray
	cvtColor(src_gau, src_gray, CV_BGR2GRAY);

	/// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Mat abs_grad_x2, abs_grad_y2;
	/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);

	convertScaleAbs(grad_x, abs_grad_x, 0.5, 128);
	convertScaleAbs(grad_x, abs_grad_x2);
	imshow("Gx", abs_grad_x);

	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(src_gray, grad_y, ddepth, 0, 1, 3, -scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y, 0.5, 128);
	convertScaleAbs(grad_y, abs_grad_y2);
	imshow("Gy", abs_grad_y);

	/// Total Gradient (approximate)

	addWeighted(abs_grad_x2, 0.5, abs_grad_y2, 0.5, 0, grad);
	imshow("Magnitud del Gradiente", grad);

	// Angle
	Mat angulo = grad_x.clone();
	for (int i = 0; i < 2 * angulo.rows * angulo.cols; i++)
	{
		double directionRAD = atan2(grad_y.data[i], grad_x.data[i]);
		angulo.data[i] = directionRAD * 128 / PI;
	}
	imshow("Angulo", angulo);
	
	//ELIMINAR RUIDO
	//Eliminar ruido desde la media
	cv::Scalar mean = cv::mean(grad);
	Mat sinruido = grad.clone();
	for (int i = 0; i <  sinruido.rows * sinruido.cols; i++)
	{
		double data = sinruido.data[i];
		if (data < mean[0]) {
			sinruido.data[i] = 0;
		}
		
	}
	imshow("Grad sin ruido", sinruido);
	Mat sinruido2;
	equalizeHist(sinruido, sinruido2);

	// Opcional aplicar otra vez pero tras ecualizar
	mean = cv::mean(sinruido2);
	
	for (int i = 0; i < sinruido2.rows * sinruido2.cols; i++)
	{
		double data = sinruido2.data[i];
		if (data < mean[0]) {
			sinruido2.data[i] = 0;
		}

	}
	imshow("Grad sin ruido2", sinruido2);
	//DETECCION DE LINEAS 
	//Mat dst, cdst;
	Mat cdst;
	//Canny(src_gray, dst, 50, 200, 3);
	//cvtColor(dst, cdst, CV_GRAY2BGR);
	cvtColor(sinruido2, cdst, CV_GRAY2BGR);

	vector<Vec4i> lines;
	HoughLinesP(sinruido2, lines, 1, CV_PI / 180, 50, 50, 10);
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, CV_AA);
	}

	imshow("source", sinruido2);
	imshow("detected lines", cdst);


	

	//Finish program
	std::cout << "Pulsa una tecla para terminar ";
	cv::waitKey(0);

	return 0;
}
