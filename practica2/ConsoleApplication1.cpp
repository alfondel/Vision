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
		//
		
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
		Moments m = moments(contours[i], false);
		int cX = int(m.m10 / m.m00);
		int cY = int(m.m01 / m.m00);
		string text = to_string(i);
		putText(drawing, text, cvPoint(cX, cY),
			FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
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



vector<vector<double>> descriptores() {
	vector<vector<double>> desc(5,vector<double>(5));
	for (size_t i = 0; i < contours.size(); i++) {

		double hu[7];
		double area = contourArea(contours[i]);
		double perimeter = arcLength(contours[i], true);
		HuMoments(moments(contours[i], false), hu);

		cout << "Objeto " << i << ", area:" << area << ", perimetro: " << perimeter << ", hu1: " << hu[0] << ", hu2: " << hu[1] << ", hu3: " << hu[2] << endl;
		if (area>100) {
			desc[i][0] = area;
			desc[i][1] = perimeter;
			desc[i][2] = hu[0];
			desc[i][3] = hu[1];
			desc[i][4] = hu[2];
		}
	}
	return desc;
}

void mahalannobis(vector<double> valores) {
	ifstream fichero;
	double minDist = 100;
	int tipo = -1;
	fichero.open("objetos.txt");
	vector<double> total(5);
	vector<double> total2(5);
	for (int tipos = 0; tipos < 5; tipos++) {
		vector<double> total(5);
		vector<double> total2(5);
		for (int i = 0; i < 5; i++) {
			fichero >> total[i];
		}
		for (int i = 0; i < 5; i++) {
			fichero >> total2[i];

		}
		double distT = 0;
		for (int i = 0; i < 5; i++) {
			distT += ((valores[i] - total[i])*(valores[i] - total[i])) / total2[i];
		}
		cout << tipos << " " << distT << endl;
		if (distT <= minDist) {
			minDist = distT;
			tipo = tipos;
		}
	}
	if (tipo == 0) {
		cout << "El objeto es de tipo circulo" << endl;
	}
	else if (tipo == 1) {
		cout << "El objeto es de tipo rueda" << endl;
	}
	else if (tipo == 2) {
		cout << "El objeto es de tipo vagon" << endl;
	}
	else if (tipo == 3) {
		cout << "El objeto es de tipo triangulo" << endl;
	}
	else if (tipo == 4) {
		cout << "El objeto es de tipo rectangulo" << endl;
	}
	else {
		cout << "El objeto es de tipo desconocido" << endl;
	}

}

int main()
{

	cout << "Umbralizando imágenes \n";
	Mat image;
	vector<vector<double>> medias(5, vector<double>(5));
	vector<vector<double>> varianzas(5, vector<double>(5));
	ofstream fichero;
	fichero.open("objetos.txt");

	for (int tipo = 0; tipo<5; tipo++) {
		vector< vector<double> > valores(5, vector<double>(5));
		vector<double> total(5);
		vector<double> total2 = { 0,0,0,0,0 };
		for (int i = 1; i < 6; i++) {
			string ruta;
			if (tipo == 0) {
				cout << "circulo ";
				ruta = "C:/Users/Alfonso/Downloads/imagenesT2/circulo" + to_string(i) + ".pgm";
			}
			else if (tipo == 1) {
				cout << "rueda ";
				ruta = "C:/Users/Alfonso/Downloads/imagenesT2/rueda" + to_string(i) + ".pgm";
			}
			else if (tipo == 2) {
				cout << "vagon ";

				ruta = "C:/Users/Alfonso/Downloads/imagenesT2/vagon" + to_string(i) + ".pgm";

			}
			else if (tipo == 3) {
				cout << "triangulo ";
				ruta = "C:/Users/Alfonso/Downloads/imagenesT2/triangulo" + to_string(i) + ".pgm";
			}
			else if (tipo == 4) {
				cout << "rectangulo ";
				ruta = "C:/Users/Alfonso/Downloads/imagenesT2/rectangulo" + to_string(i) + ".pgm";
			}
			image = imread(ruta, CV_LOAD_IMAGE_COLOR);
			umbralizarOTSU(image);
			dilation(image, image, 0, 6);
			erosion(image, image, 0, 6);
			contornos(image);
			vector<double> desc = descriptores()[0];
			valores[i - 1] = desc;
			for (int j = 0; j < 5; j++) {
				total[j] += desc[j];
			}

		}
		for (int j = 0; j < 5; j++) {
			total[j] = total[j] / 5;
		}

		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 5; j++) {
				total2[j] += (valores[i][j] - total[j])*(valores[i][j] - total[j]);
			}
		}
		for (int j = 0; j < 5; j++) {
			total2[j] = total2[j] / 4;
		}


		fichero << total[0] << " " << total[1] << " " << total[2] << " " << total[3] << " " << total[4] << endl;
		fichero << total2[0] << " " << total2[1] << " " << total2[2] << " " << total2[3] << " " << total2[4] << endl;
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	string ruta;
	ruta = "C:/Users/Alfonso/Downloads/imagenesT2/reco3.pgm";
	image = imread(ruta, CV_LOAD_IMAGE_COLOR);
	imshow("Imagen", image);
	umbralizarOTSU(image);
	dilation(image, image, 0, 6);
	erosion(image, image, 0, 6);
	contornos(image);

	vector<vector<double>> desc = descriptores();
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cout <<endl<< "Reconocimiento con distancia de mahalannobis"<<endl;
	for (int i = 0; i < desc.size(); i++) {
		cout << "Objeto " + to_string(i) << endl;
		mahalannobis(desc[i]);
		cout << endl ;
	}
	

	waitKey(0);

	return 0;
}
