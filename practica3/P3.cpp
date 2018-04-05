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

void votar_recta(float tetha, float ro, int cols, vector<int> votos) {

	int corte = ro / cosf(tetha);		// Se calcula el corte con el eje X. 
	if (corte < cols / 2 && corte >= -cols / 2) {	// Se comprueba que corta en la imagen.
		corte = corte + cols / 2;		// Se pone el corte en el rango.
		votos[corte] = votos[corte] + 1;	// Se actualiza el valor.
	}
}

int votar_rectas(Vec4i linea) {
	int x1 = linea[0];
	int y1 = linea[1];
	int x2 = linea[2];
	int y2 = linea[3];
	int x3 = 0;
	int y3 = 225;
	int x4 = 500;
	int y4 = 225;
	if (((x1 - x2)*(y3 - y4)) - ((y1 - y2)*(x3 - x4)) != 0) {
		int pos = (((x1*y2 - y1 * x2)*(x3 - x4)) - ((x1 - x2)*(x3*y4 - y3 * x4))) / (((x1 - x2)*(y3 - y4)) - ((y1 - y2)*(x3 - x4)));
		if (pos >= 0 && pos<500) {
			return pos;
		}
	}
	return -1;
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
	Mat src, src_gray, src_gau;
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
	for (int i = 0; i < sinruido.rows * sinruido.cols; i++)
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
	Mat cdst,cannyout, cannyresult;
	Canny(src_gray, cannyout, 50, 200, 3);
	cvtColor(cannyout, cannyresult, CV_GRAY2BGR);
	cvtColor(sinruido2, cdst, CV_GRAY2BGR);
	
	vector<Vec4i> lines;
	//HoughLinesP(sinruido2, lines, 1, CV_PI / 180, 50, 50, 10);
	HoughLinesP(cannyout, lines, 1, CV_PI / 180, 50, 50, 10);
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, CV_AA);
	}

	
	Mat cdst2 = cdst.clone();
	Mat cdst3 = cdst.clone();
	imshow("source", sinruido2);
	imshow("detected lines", cdst);
	
	float umbral = 80;
	vector<int> votos2(500);
	//VOTACION with contour pixels
	for (int i = 0; i < cdst.rows; i++) {
		for (int j = 0; j < cdst.cols;j++) {
			if (grad.at<uchar>(i,j)>umbral) {
				float tetha = angulo.data[j + (i*angulo.cols)];
				//float ro = x * cosf(tetha) + y * sinf(tetha);

				int x1 = j;
				int y1 = i;
				int x2 = 10 * cosf(tetha);
				int y2 = 10 * sinf(tetha);
				Vec4i linea = Vec4i(x1,y1,x2,y2 );
				int voto = votar_rectas(linea);
				if (voto > -1) {
					votos2[voto] = votos2[voto] + 1;
				}
			}
		}
	}
	//VOTACION WITH HoughLinesP
	vector<int> votos(500);
	for (Vec4i linea : lines) {
		int voto = votar_rectas(linea);
		if (voto > -1) {
			votos[voto] = votos[voto] + 1;
		}
	}

	//Contar votos de la primera forma
	int maxvotes = 0;
	int indice = 0;
	for (int i = 0; i < votos.size(); i++) {
		double voto = votos.at(i);
		//cout << i <<":" << voto <<"\t";
		//paint on cst image the points of the horizon
		if (voto > 0) {
			circle(cdst,Point(i, 225), voto,Scalar(255, 0, 0),-1,8);
		}
		
		if (voto > maxvotes) {
			//cout << "votos " << votos.at(i) << endl;
			maxvotes = voto;
			indice = i;
		}
	}
	circle(cdst, Point(indice, 225), maxvotes, Scalar(0, 255, 0), -1, 8);
	imshow("detected lines with points on horizon", cdst);
	cout <<"\n"<< maxvotes<<" votes at indice " << indice <<endl;

	//Contar votos con el contorno
	maxvotes = 0;
	indice = 0;
	for (int i = 0; i < votos2.size(); i++) {
		double voto = votos2.at(i);
		cout << i << ":" << voto << "\t";
		//paint on cst image the points of the horizon
		if (voto > 0) {
			circle(cdst2, Point(i, 225), voto, Scalar(255, 0, 0), -1, 8);
		}

		if (voto > maxvotes) {
			//cout << "votos " << votos.at(i) << endl;
			maxvotes = voto;
			indice = i;
		}
	}
	circle(cdst2, Point(indice, 225), maxvotes, Scalar(0, 255, 0), -1, 8);
	imshow("detected lines with points on horizon (contour version)", cdst2);
	cout << "\n" << maxvotes << " votes at indice " << indice << endl;


	//Contar votos con el contorno de canny
	//VOTACION with contour pixels de canny
	vector<int> votos3(500);
	for (int i = 0; i < cannyout.rows; i++) {
		for (int j = 0; j < cannyout.cols; j++) {
			if (cannyout.at<uchar>(i, j)>umbral) {
				
				float tetha = angulo.data[j + (i*angulo.cols)];
				//float ro = x * cosf(tetha) + y * sinf(tetha);
				int x1 = j;
				int y1 = i;
				int x2 = 10 * cosf(tetha);
				int y2 = 10 * sinf(tetha);
				Vec4i linea = Vec4i(x1, y1, x2, y2);
				int voto = votar_rectas(linea);
				if (voto > -1) {
					votos3[voto] = votos3[voto] + 1;
				}
			}
		}
	}
	
	maxvotes = 0;
	indice = 0;
	for (int i = 0; i < votos3.size(); i++) {
		double voto = votos3.at(i);
		cout << i << ":" << voto << "\t";
		//paint on cst image the points of the horizon
		if (voto > 0) {
			circle(cdst3, Point(i, 225), voto, Scalar(255, 0, 0), -1, 8);
		}

		if (voto > maxvotes) {
			//cout << "votos " << votos.at(i) << endl;
			maxvotes = voto;
			indice = i;
		}
	}
	circle(cdst3, Point(indice, 225), maxvotes, Scalar(0, 255, 0), -1, 8);
	imshow("detected lines with points on horizon (contour version of canny)", cdst3);
	cout << "\n" << maxvotes << " votes at indice " << indice << endl;
	//Finish program
	std::cout << "Pulsa una tecla para terminar ";
	cv::waitKey(0);

	return 0;
}