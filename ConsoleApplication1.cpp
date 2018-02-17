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
		double a = 2; //contraste 
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

int main()
{
	histograma();
	/*VideoCapture stream1(0);
	if (!stream1.isOpened()) {
	cout << "Camara no accesible";
	}
	int a = 2;
	int b = 10;
	stream1.open(0);
	while (true) {

	Mat cameraFrame;
	stream1.read(cameraFrame);
	if (!stream1.read(cameraFrame)) {
	cout << "Camara nula \n";
	}
	imshow("Cam Original", cameraFrame);
	cameraFrame = cameraFrame * a;
	cameraFrame = cameraFrame + b;
	imshow("Cam Con Contraste", cameraFrame);
	if (waitKey(30)>=0) {
	break;
	}
	}

	stream1.release();
	destroyAllWindows();*/
	return 0;
}


