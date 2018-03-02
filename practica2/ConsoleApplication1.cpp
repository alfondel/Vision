// ConsoleApplication1.cpp: define el punto de entrada de la aplicación de consola.

//Alfonso Delgado Vellosillo - 679745
//Daniel Martinez Martinez - 538798


#include "stdafx.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds

using namespace cv;
using namespace std; 

//Umbralización en hsv
void umbralizar(Mat imagen) {
	MatIterator_<Vec3b> it, end;
	Mat hsv;
	cvtColor(imagen, hsv, CV_BGR2HSV);
	for (it = hsv.begin<Vec3b>(), end = hsv.end<Vec3b>(); it != end; ++it) {
		if (140 < (*it)[2] ) {
			(*it)[2] = 255;
		}
		else {
			(*it)[2] = 0;
		}

	}
	cvtColor(hsv, imagen, CV_HSV2BGR);
	imshow("Imagen umbralizada", imagen);
}
//umbralizacion en OTSU
void umbralizarOTSU(Mat imagen) {
	Mat gray;
	cvtColor(imagen, gray, CV_BGR2GRAY);
	Mat umb;
	threshold(gray, umb, 0, 255, CV_THRESH_OTSU);
	cvtColor(umb, imagen, CV_GRAY2BGR);
	imshow("Imagen umbralizada", imagen);

}

void umbralizarAdap(Mat imagen) {
	Mat gray;
	cvtColor(imagen, gray, CV_BGR2GRAY);
	Mat umb;
	adaptiveThreshold(gray, umb, 255, ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,11 ,2);
	imshow("Imagen umbralizada", umb);

}

int main()
{
	cout << "Umbralizando imágenes \n";
	Mat image;
	String ruta("C:/Users/Alfonso/Downloads/imagenesT2/vagon5.pgm");
	image = imread(ruta, CV_LOAD_IMAGE_COLOR); 
	imshow("Imagen",image);
	umbralizarAdap(image);
	waitKey(0);

	return 0;
}


