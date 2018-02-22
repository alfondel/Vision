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
	Mat src, dst,contraste,ambasr,dst_hsv;

	while (true)
	{
		/// Load image
		stream1.read(src);
		stream1.read(src);

		if (src.data == NULL)
		{
			return -1;
		}
		vector<Mat> hsv_planes;
		Mat HSV;
		cvtColor(src, HSV, CV_BGR2HSV);
		split(HSV, hsv_planes);
		Mat h = hsv_planes[0]; // H channel
		Mat s = hsv_planes[1]; // S channel
		Mat v = hsv_planes[2]; // V channel
		//procedemos a ecualizar el histograma
		//esta funcion ya calcula el histograma y lo ecualiza 
		equalizeHist(hsv_planes[2], hsv_planes[2]);


		//contraste y brillo
		double a = 1.4; //contraste 
		double b = 10; // brillo
		vector<Mat> colores2;
		split(src, colores2);
		colores2[0] = colores2[0] * a + b;
		colores2[1] = colores2[1] * a + b;
		colores2[2] = colores2[2] * a + b;
		//primero contraste y luego ecu
		/*vector<Mat> ambr;
		split(src, ambr);
		
		ambr[0] = ambr[0] * a + b;
		ambr[1] = ambr[1] * a + b;
		ambr[2] = ambr[2] * a + b;

		equalizeHist(ambr[0], ambr[0]);
		equalizeHist(ambr[1], ambr[1]);
		equalizeHist(ambr[2], ambr[2]);*/
		
		//recomponemos las imagenes
		merge(hsv_planes, dst_hsv);
		cvtColor(dst_hsv, dst, cv::COLOR_HSV2BGR);
		merge(colores2, contraste);
		//merge(ambr, ambasr);
		//pintamos los histogramas
		pintarHistograma(src,"Sin Ecu");
		pintarHistograma(dst,"Con Ecu");
		pintarHistograma(contraste, "Solo constranste y brillo");
		//pintarHistograma(ambasr, "contraste,brillo y ecualizacion");
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
		Mat imageR;
		Mat imageG;
		Mat imageB;
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
					/*double c = (*it)[0] + 50;
					if (c > 255) {
						c = 255;
					}
					(*it)[0] = c;
				//pasar a color rojo
					double c = (*it)[2] + 50;
					if (c > 255) {
						c = 255;
					}
					(*it)[2] = c;*/
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
		Mat centroids;
		Mat label;
		stream1.read(image);
		stream1.read(image);

		Mat aux = Mat(image.cols*image.rows, 3, CV_32F);

		imshow("Imagen sin procesar", image);

		//transformamos la matriz a una matriz de floats de floats, ya que kmeans solo reconoce un vector de floats 
		MatIterator_<Vec3b> it, end;
		int i = 0; 
		for (it = image.begin<Vec3b>(), end = image.end<Vec3b>(); it != end; ++it) {
			aux.at<float>(i, 0) = (*it)[0];
			aux.at<float>(i, 1) = (*it)[1];
			aux.at<float>(i, 2) = (*it)[2];
			i++;
		}
		//Aplicamos kmeans 
		kmeans(aux,numColores,label,TermCriteria(), 1, KMEANS_RANDOM_CENTERS ,centroids);
		
		//seleccionamos el centroide asignado para cada pixel almacenado en label, y lo retransformamos a un Vect3b en vez de float
		i = 0;
		for (it = image.begin<Vec3b>(), end = image.end<Vec3b>(); it != end; ++it) {
			//en label guardamos de cada pixel a que centroide pertence 
			int posicion = label.at<int>(i, 0);
			(*it)[0] = centroids.at<float>(posicion, 0);
			(*it)[1] = centroids.at<float>(posicion, 1);
			(*it)[2] = centroids.at<float>(posicion, 2);
			i++;
		}
		imshow("Imagen procesada", image);
		if (waitKey(30) >= 0) {
			break;
		}
	}

	
}

void distorsionRadial() {
	VideoCapture stream1(0);
	Mat src;
	Mat dst;
	Mat mapx;
	Mat mapy;
	double k1 = 0.000002;
	double k2 = 0.00000000001;

	stream1.read(src);
	stream1.read(src);
	stream1.read(dst);
	stream1.read(dst);
	mapx.create(src.size(), CV_32FC1);
	mapy.create(src.size(), CV_32FC1);
	double centery = src.rows / 2;
	double centerx = src.cols / 2;

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			// calculate radio
			double x = j - centerx;
			double y = centery - i;

			double rr = x * x + y * y;
			double cons = (1 + k1 * rr + k2 * rr*rr);
			int xcorr = (x * cons) + centerx;
			int ycorr = centery - (y * cons);
			if (xcorr >= src.cols) {
				xcorr = src.cols - 1;
			}
			if (xcorr < 0) {
				xcorr = 0;
			}
			if (ycorr >= src.rows) {
				ycorr = src.rows - 1;
			}
			if (ycorr < 0) {
				ycorr = 0;
			}
			mapx.at<float>(ycorr, xcorr) = (float)j;
			mapy.at<float>(ycorr, xcorr) = (float)i;

		}
	}
	while (true) {
		/// Load image
		stream1.read(src);

		imshow("Imagen sin procesar", src);

		remap(src, dst, mapx, mapy, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
		imshow("Imagen procesada", dst);
		if (waitKey(30) >= 0) {
			break;
		}
	}
}

void distorsion() {

	Mat image;
	VideoCapture stream1(0);
	int k1=2;

	Mat map_x, map_y, output;

	while (true) {
		stream1.read(image);
		stream1.read(image);
		double Cy = (double)image.cols / 2;
		double Cx = (double)image.rows / 2;
		map_x.create(image.size(), CV_32FC1);
		map_y.create(image.size(), CV_32FC1);

		for (int x = 0; x < map_x.rows; x++) {
			for (int y = 0; y < map_y.cols; y++) {
				double r2 = (x - Cx)*(x - Cx) + (y - Cy)*(y - Cy);
				map_x.at<float>(x, y) = (double)((y - Cy) / (1 + double(k1 / 1000000.0)*r2) + Cy); // se suma para obtener la posicion absoluta
				map_y.at<float>(x, y) = (double)((x - Cx) / (1 + double(k1 / 1000000.0)*r2) + Cx); // la posicion relativa del punto al centro
			}
		}
		remap(image, output, map_x, map_y, CV_INTER_LINEAR);
		imshow("Imagen procesada", output);
		if (waitKey(30) >= 0) {
			break;
		}
	}
}

int main()
{
	cout << "Efecto contraste: Presiona esc para continuar\n";
	histograma();
	destroyAllWindows();
//	cout << "Efecto alien: Presiona esc para continuar\n";
//	efectoAlien();
//	destroyAllWindows();
//	cout << "Reducir colores: Presiona esc para continuar\n";
//	reducirColores(25);
//	destroyAllWindows();
//	distorsionRadial();
//	cout << "Reducir colores: Presiona esc para continuar\n";
//	destroyAllWindows();
	return 0;
}


