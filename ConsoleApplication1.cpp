// ConsoleApplication1.cpp: define el punto de entrada de la aplicación de consola.

#include "stdafx.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std; 

void pintarHistograma(Mat imagen,String nombre) {
	Mat b_hist, g_hist, r_hist;
	/// Separate the image in 3 places ( B, G and R )
	vector<Mat> bgr_planes;
	vector<Mat> bgr_planesEqu;
	//es la funcion split la que separa la imagen en colores 
	split(imagen, bgr_planes);

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	/// Compute the histograms:
	//la funcion calcHist calcula el histograma autormaticamente
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}

	/// Display
	namedWindow("Histograma"+nombre, CV_WINDOW_AUTOSIZE);
	imshow("Histograma"+nombre, histImage);
	imshow("Imagen" + nombre, imagen);
}


int histograma() {

	VideoCapture stream1(0);
	Mat src, dst,contraste,ambasr;

	while (true)
	{
		/// Load image
		stream1.read(src);
		stream1.read(src);

		if (src.data == NULL)
		{
			return -1;
		}
		vector<Mat> colores;
		//Separamos la imagen en colores
		split(src, colores);
		//procedemos a ecualizar el histograma
		//esta funcion ya calcula el histograma y lo ecualiza 
		equalizeHist(colores[0], colores[0]);
		equalizeHist(colores[1], colores[1]);
		equalizeHist(colores[2], colores[2]);
		//contraste y brillo
		double a = 1.4; //contraste 
		double b = 10; // brillo
		vector<Mat> colores2;
		split(src, colores2);
		colores2[0] = colores2[0] * a + b;
		colores2[1] = colores2[1] * a + b;
		colores2[2] = colores2[2] * a + b;
		//primero contraste y luego ecu
		vector<Mat> ambr;
		split(src, ambr);
		ambr[0] = ambr[0] * a + b;
		ambr[1] = ambr[1] * a + b;
		ambr[2] = ambr[2] * a + b;
		equalizeHist(ambr[0], ambr[0]);
		equalizeHist(ambr[1], ambr[1]);
		equalizeHist(ambr[2], ambr[2]);
		
		//recomponemos las imagenes
		merge(colores, dst);
		merge(colores2, contraste);
		merge(ambr, ambasr);
		//pintamos los histogramas
		pintarHistograma(src,"Sin Ecu");
		pintarHistograma(dst,"Con Ecu");
		pintarHistograma(contraste, "Solo constranste y brillo");
		pintarHistograma(ambasr, "contraste,brillo y ecualizacion");
		if (waitKey(30) >= 0) {
			break;
		}
	}
	return 0;


}

void efectoAlien() {
	VideoCapture stream1(0);
	//Creamos la tabla para efectos de tipo alien
	//Relacion de los colores para el color de piel rosa claro
	double relacionRedGreen = 1.16;
	double relacionRedBlue = 1.32;
	double relacionGreenBlue = 1.15;
	double ratioparecido = 0.85;

	while (true) {
		
		//leer la imagen de la camara
		Mat image;
		stream1.read(image);
		stream1.read(image);
		imshow("Imagen sin procesar", image);
		//Procesarla
		MatIterator_<Vec3b> it, end;
		for (it = image.begin<Vec3b>(), end = image.end<Vec3b>(); it != end; ++it){

			if ((*it)[1] != 0 && (*it)[2] != 0) {

				double red = (*it)[2];
				double green = (*it)[1];
				double blue = (*it)[0];

				double rg = red / green; //relacion en el pixel de rojo cn verde
				double rb = red / blue; //relacion en el pixel de rojo cn azul
				double gb = green / blue; //relacion en el pixel de verde con azul

				if (abs(rg / relacionRedGreen) >= ratioparecido && abs(rg / relacionRedGreen) >= ratioparecido && abs(rg / relacionRedGreen) >= ratioparecido) {
				//Pasar a color verde
					double c = (*it)[1] + 50;
					if (c > 255) {
						c = 255;
					}
					(*it)[1] = c;

				//pasar a color azul
					//(*it)[0] = (*it)[0] + 50;
				//pasar a color rojo
					//(*it)[2] = (*it)[2] + 50;
				}
			}
		}
		
		//pintamos las imagenes
		imshow("Imagen procesada", image);
		if (waitKey(30) >= 0) {
			break;
		}
	}
	
}

void reducirColores(int numColores) {

	vector<Vec3b> centroids;
	VideoCapture stream1(0);
	while (true) {
		Mat image;
		stream1.read(image);
		stream1.read(image);
		imshow("Imagen sin procesar", image);
		for (int i = 0; i < numColores;i++) {

		}
		Mat_<Vec3b>::iterator it = image.begin<Vec3b>();
		Mat_<Vec3b>::iterator itend = image.begin<Vec3b>();
		double max;
		for (; it != itend; ++it) {
			//calcular valores del pixel 
			for (Vec3b centroide : centroids) {
				//compararlos con los centroides 

				//quedarnos con el maximo
			}
			//sustituir valor del pixel por el del centroide 
		}
	}
}

int main()
{
//	cout << "Efecto contraste: Presiona esc para continuar\n";
//	histograma();
//	destroyAllWindows();
//	cout << "Efecto alien: Presiona esc para continuar\n";
	efectoAlien();
	cout << "Reducir colores: Presiona esc para continuar\n";
	//reducirColores(25);
	return 0;
}


