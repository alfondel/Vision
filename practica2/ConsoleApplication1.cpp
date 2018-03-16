// ConsoleApplication1.cpp: define el punto de entrada de la aplicación de consola.

//Alfonso Delgado Vellosillo - 679745
//Daniel Martinez Martinez - 538798


#include "stdafx.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include <vector>
using namespace cv;
using namespace std;

vector<vector<Point> > contours;

//Umbralización en hsv
void umbralizar(Mat imagen) {
	MatIterator_<Vec3b> it, end;
	Mat hsv;
	cvtColor(imagen, hsv, CV_BGR2HSV);
	for (it = hsv.begin<Vec3b>(), end = hsv.end<Vec3b>(); it != end; ++it) {
		if (140 < (*it)[2]) {
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
	MatIterator_<Vec3b> it, end;
	for (it = imagen.begin<Vec3b>(), end = imagen.end<Vec3b>(); it != end; ++it) {
		if ((*it)[2] == 255) {
			(*it)[0] = 0;
			(*it)[1] = 0;
			(*it)[2] = 0;
		}
		else {
			(*it)[0] = 255;
			(*it)[1] = 255;
			(*it)[2] = 255;
		}

	}
	imshow("Imagen umbralizada", imagen);

}

void umbralizarAdap(Mat imagen) {
	Mat gray;
	cvtColor(imagen, gray, CV_BGR2GRAY);
	Mat umb;
	adaptiveThreshold(gray, umb, 255, ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 11, 2);
	imshow("Imagen umbralizada", umb);

}

//Lo he sacado de un tutorial de la funcion que nos piden que usemos
void contornos(Mat imagen) {
	int thresh = 100;
	RNG rng(12345);
	Mat gray;
	Mat canny_output;

	vector<Vec4i> hierarchy;
	cvtColor(imagen, gray, CV_BGR2GRAY);
	/// Detect edges using canny
	Canny(gray, canny_output, thresh, thresh * 2, 3);
	/// Find contours
	findContours(canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
	}

	/// Show in a window
	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	imshow("Contours", drawing);

}


void erosion(Mat src, Mat erosion_dst, int erosion_elem, int erosion_size)
{
	int erosion_type;
	if (erosion_elem == 0) { erosion_type = MORPH_RECT; }
	else if (erosion_elem == 1) { erosion_type = MORPH_CROSS; }
	else if (erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

	Mat element = getStructuringElement(erosion_type,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(erosion_size, erosion_size));

	/// Apply the erosion operation
	erode(src, erosion_dst, element);
	imshow("Erosion Demo", erosion_dst);
}


void dilation(Mat src, Mat dilation_dst, int dilation_elem, int dilation_size)
{
	int dilation_type;
	if (dilation_elem == 0) { dilation_type = MORPH_RECT; }
	else if (dilation_elem == 1) { dilation_type = MORPH_CROSS; }
	else if (dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

	Mat element = getStructuringElement(dilation_type,
		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(dilation_size, dilation_size));
	/// Apply the dilation operation
	dilate(src, dilation_dst, element);
	imshow("Dilation Demo", dilation_dst);
}



vector<double> descriptores() {
	vector<double> desc(5);
	for (size_t i = 0; i < contours.size(); i++) {

		double hu[7];
		double area = contourArea(contours[i]);
		double perimeter = arcLength(contours[i], true);
		HuMoments(moments(contours[i], false), hu);
		
		cout << "Objeto " << i << ", area:" << area << ", perimetro: " << perimeter << ", hu1: " << hu[0] << ", hu2: " << hu[1] << ", hu3: " << hu[2] << endl;
		if (area>2) {
			desc[0] = area;
			desc[1] = perimeter;
			desc[2] = hu[0];
			desc[3] = hu[1];
			desc[4] = hu[2];
		}
	}
	return desc;
}

void mahalannobis(vector<double> total, vector<double> total2, vector<double> valores) {
	vector<double> dist(5);
	double distT = 0;
	for (int i = 0; i < 5;i++) {
		dist[i] = ((valores[i] - total[i])*(valores[i] - total[i])) / total2[i];
		distT += dist[i];
	}
	cout << distT << endl;
}

int main()
{

	cout << "Umbralizando imágenes \n";
	Mat image;
	vector<double> total(5);

	vector< vector<double> > valores(5, vector<double>(5));
	for (int i = 1; i < 6;i++) {
		string ruta;
		ruta="C:/Users/Alfonso/Downloads/imagenesT2/circulo"+to_string(i)+".pgm";
		image = imread(ruta, CV_LOAD_IMAGE_COLOR);
		imshow("Imagen", image);
		umbralizarOTSU(image);
		dilation(image, image, 2, 4);
		erosion(image, image, 2, 4);
		contornos(image);
		vector<double> desc = descriptores();
		valores[i-1] = desc;
		for (int j = 0; j < 5;j++) {
			total[j] += desc[j];
		}

	}
	for (int j = 0; j < 5; j++) {
		total[j] = total[j]/5;
	}
	vector<double> total2 = { 0,0,0,0,0 };
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5;j++) {
			total2[j] += (valores[i][j] - total[j])*(valores[i][j] - total[j]);
		}
	}
	for (int j = 0; j < 5; j++) {
		total2[j] = total2[j] / 4;

	}
	cout << "areaV:" << total2[0] << ", perimetroV: " << total2[1] << ", hu1V: " << total2[2] << ", hu2V: " << total2[3] << ", hu3V: " << total2[4] << endl;
	cout << "areaT:" << total[0] << ", perimetroT: " << total[1] << ", hu1T: " << total[2] << ", hu2T: " << total[3]<< ", hu3T: " << total[4] << endl;

	ofstream fichero;
	fichero.open("objetos.txt");
	fichero <<  total2[0] << " " << total2[1] << " " << total2[2] << " " << total2[3] << " " << total2[4] << endl;
	fichero << total[0] << " " << total[1] << " " << total[2] << " " << total[3] << " " << total[4] << endl;


	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	string ruta;
	ruta = "C:/Users/Alfonso/Downloads/imagenesT2/vagon2.pgm";
	image = imread(ruta, CV_LOAD_IMAGE_COLOR);
	imshow("Imagen", image);
	umbralizarOTSU(image);
	dilation(image, image, 2, 4);
	erosion(image, image, 2, 4);
	contornos(image);
	vector<double> desc = descriptores();
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	mahalannobis(total, total2,desc);

	waitKey(0);

	return 0;
}

