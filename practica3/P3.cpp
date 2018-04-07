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

int votar_recta(float tetha, float ro, int cols) {

	int corte = ro / cosf(tetha);		// Se calcula el corte con el eje X. 
	if (corte < cols / 2 && corte >= -cols / 2) {	// Se comprueba que corta en la imagen.
		corte = corte + (cols / 2);		// Se pone el corte en el rango.
		return corte;
	}
	return -1;
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
			return pos / 10;
		}
	}
	return -1;
}

void pintarX(int x, Mat mat) {
	int y = mat.cols / 2;
	Point p1 = Point(x, y - 10);
	Point p2 = Point(x, y + 10);
	Point p3 = Point(x - 10, y);
	Point p4 = Point(x + 10, y);

	// Se dibuja la X.
	line(mat, p1, p2, CV_RGB(255, 0, 0), 4);
	line(mat, p3, p4, CV_RGB(255, 0, 0), 4);
	circle(mat, Point(x, y), 1, CV_RGB(255, 0, 0), 3);
	imshow("Punto de fuga", mat);
}

void gradients(cv::Mat &src_gray, cv::Mat &grad_x, cv::Mat &grad_y, cv::Mat &grad)
{
	int scale = 1;
	int delta = 0;
	int ddepth = CV_32F;
	// Gradient X
	Mat abs_grad_x;
	Mat abs_grad_x2;
	Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x, 0.5, 128);
	convertScaleAbs(grad_x, abs_grad_x2);
	imshow("Gx", abs_grad_x);
	// Gradient Y
	Mat abs_grad_y;
	Mat abs_grad_y2;
	Sobel(src_gray, grad_y, ddepth, 0, 1, 3, -scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y, 0.5, 128);
	convertScaleAbs(grad_y, abs_grad_y2);
	imshow("Gy", abs_grad_y);
	// Total Gradient (approximate)
	addWeighted(abs_grad_x2, 0.5, abs_grad_y2, 0.5, 0, grad);
}

void reduceNoisy(cv::Mat &grad, cv::Mat &sinruido)
{
	cv::Scalar mean = cv::mean(grad);
	Mat aux = grad.clone();
	for (int i = 0; i < aux.rows * aux.cols; i++) {
		double data = aux.data[i];
		if (data < mean[0]) {
			aux.data[i] = 0;
		}
	}
	imshow("Grad sin ruido", aux);

	equalizeHist(aux, sinruido);
	mean = cv::mean(sinruido);
	threshold(sinruido, sinruido, mean[0], 255, 0);
}
void fullLine(cv::Mat *img, cv::Point a, cv::Point b, cv::Scalar color) {
	double deno = (b.x - a.x);
	if (deno != 0) {
		double slope = (b.y - a.y) / deno;
		Point p(0, 0), q(img->cols, img->rows);

		p.y = -(a.x - p.x) * slope + a.y;
		q.y = -(b.x - q.x) * slope + b.y;

		line(*img, p, q, color, 1, 8, 0);
	}

}
int main(int argc, char * argv[]) {

	// Read image from parameters
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

	// Load an image in src
	Mat src;
	src = imread(file);
	if (!src.data) {
		return -1;
	}

	// Apply gaussian blur to image in src_gau
	Mat src_gau;
	int sigma = 3;
	GaussianBlur(src, src_gau, Size(sigma, sigma), 0, 0, BORDER_DEFAULT);

	// Convert it to gray src_gray
	Mat src_gray;
	cvtColor(src_gau, src_gray, CV_BGR2GRAY);

	// Generate grad_x and grad_y
	Mat grad_x;
	Mat grad_y;
	Mat grad;
	gradients(src_gray, grad_x, grad_y, grad);
	Mat angulo = grad.clone();
	cartToPolar(grad_x, grad_y, grad, angulo,false);
	//imshow("Magnitud del Gradiente", grad/255);

	// Angle
	
	/*for (int i = 0; i < angulo.rows * angulo.cols; i++) {
		double y = grad_y.data[i];
		double x = grad_x.data[i];
		double directionRAD = atan2(y, x);
		angulo.data[i] = directionRAD * 128 / PI;
	}*/
	
	imshow("Angulo", (angulo * 128 / PI)/255);
	waitKey(0);
	// Reduce noisy from sobel
	Mat sinruido;
	reduceNoisy(grad, sinruido);
	imshow("Contorno", sinruido);

	// Detect lines 
	Mat cdst = grad;
	//cvtColor(sinruido, cdst, CV_GRAY2BGR);

	float umbral = 100;
	vector<int> votos2(50);
	//VOTACION with contour pixels
	for (int i = 0; i < cdst.rows; i++) {
		for (int j = 0; j < cdst.cols; j++) {
			if (grad.at<uchar>(i, j)> umbral) {
				float tetha = angulo.data[j + (i*angulo.cols)] * PI / 128;
				//utilizamos dist para medir la distancia a los ejes X e Y y comprobar si una recta es vertical u horizontal
				float dist = cosf(tetha);
				if (dist < 0) {
					dist = -dist;
				}
				if (dist>0.1 && dist<0.9) {
					int x1 = j;
					int y1 = i;
					int x2 = x1 - 10 * sinf(tetha);
					int y2 = y1 - 10 * cosf(tetha);
					Vec4i linea = Vec4i(x1, y1, x2, y2);
					int voto = votar_rectas(linea);
					if (voto > -1) {
						votos2[voto] = votos2[voto] + 1;
						fullLine(&cdst, Point(x1, y1), Point(x2, y2), Scalar(255, 0, 255));
						circle(cdst, Point(x1,y1), 4, Scalar(0, 0, 255), -1, 8);
						circle(cdst, Point(x2, y2), 4, Scalar(0, 255, 0), -1, 8);
						/*cout << dist << endl;
						imshow("detected lines with points on horizon (contour version)", cdst);
						waitKey(0);*/
					}
				}
			}
		}
	}
	int maxvotes = 0;
	int indice = 0;

	//Contar votos con el contorno
	maxvotes = 0;
	indice = 0;
	for (int i = 0; i < votos2.size(); i++) {
		double voto = votos2.at(i);
		cout << i << ":" << voto << "\t";
		//paint on cst image the points of the horizon
		if (voto > 0) {
			circle(cdst, Point(i * 10, 225), log(voto*voto), Scalar(255, 0, 0), -1, 8);
		}

		if (voto > maxvotes) {
			maxvotes = voto;
			indice = i * 10;
		}
	}
	circle(cdst, Point(indice, 225), log(maxvotes*maxvotes), Scalar(0, 255, 0), -1, 8);
	imshow("detected lines with points on horizon (contour version)", cdst);
	pintarX(indice, src);
	cout << "\n" << maxvotes << " votes at indice " << indice << endl;

	//Finish program
	std::cout << "Pulsa una tecla para terminar ";
	cv::waitKey(0);

	return 0;
}

