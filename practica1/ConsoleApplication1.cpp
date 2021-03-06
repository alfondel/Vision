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
	namedWindow("Histograma "+nombre, CV_WINDOW_AUTOSIZE);
	imshow("Histograma "+nombre, histImage);
	imshow("Imagen " + nombre, imagen);
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

		//recomponemos las imagenes
		merge(hsv_planes, dst_hsv);
		cvtColor(dst_hsv, dst, cv::COLOR_HSV2BGR);
		merge(colores2, contraste);
		//pintamos los histogramas
		pintarHistograma(src,"Original");
		pintarHistograma(dst,"Ecualizar brillo v en formato hsv");
		pintarHistograma(contraste, "Contraste y brillo mediante a*x+b en rgb");
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
		Mat verde;
		Mat ycrcbmask;
		Mat YCrCb;
		stream1.read(image);
		stream1.read(image);
		verde = image.clone();
		
		cvtColor(image, YCrCb, CV_BGR2YCrCb);
		
		inRange(YCrCb, Scalar(0, 137, 77), Scalar(255, 173, 127), ycrcbmask);
		
		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				
				if (ycrcbmask.at<uchar>(i, j) != 0) {

					double c = verde.at<Vec3b>(i, j)[1] + 50;
					if (c > 255) {
						c = 255;
					}
					
					verde.at<Vec3b>(i, j)[1] = c;
					
				}
			}
		}
		
		
		imshow("Color detectado YCrCb", ycrcbmask);
		imshow("Composicion YCrCb", verde);
		
		imshow("Imagen Original", image);

		
		MatIterator_<Vec3b> it, end;
		for (it = image.begin<Vec3b>(), end = image.end<Vec3b>(); it != end; ++it){
			
			//Con RGB: Procesarla mediante relaciones entre los colores para detectar el carne
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
		imshow("Imagen con deteccion del color en RGB", image);
		if (waitKey(30) >= 0) {
			break;
		}
	}
	
}

void reducirColores(int numColores) {

	vector<Vec3b> centroids;
	VideoCapture stream1(0);
	Mat image;
	Mat finalimage;
	
	Mat label;
	TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER,3,1.0);
	while (true) {
		
		stream1.read(image);
		stream1.read(image);
		Mat centroids;
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
		
		kmeans(aux,numColores,label, criteria, 1, KMEANS_RANDOM_CENTERS ,centroids);
		
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
	Mat dst2;
	Mat mapx;
	Mat mapy;
	Mat mapx2;
	Mat mapy2;
	double k1 = 0.0000029;
	double k2 = 0.00000000001;
	
	
	stream1.read(src);
	stream1.read(src);

	dst = src.clone();
	dst2 = src.clone();
	mapx.create(src.size(), CV_32FC1);
	mapy.create(src.size(), CV_32FC1);
	mapx2.create(src.size(), CV_32FC1);
	mapy2.create(src.size(), CV_32FC1);
	double centery = src.rows / 2;
	double centerx = src.cols / 2;

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			// calculate radio
			double x = j - centerx;
			double y = i - centery;
			double rr = x * x + y * y;
			double r4 = x * x * x * x + y * y * y * y;
			double cons = (1 + k1 * rr + k2 * r4);
			int xcorr =  (x / cons) + centerx;
			int ycorr =  (y / cons) + centery;

			mapx.at<float>(i,j) = (float)xcorr;
			mapy.at<float>(i,j) = (float)ycorr;

			cons = (1 + (-k1) * rr + (-k2) * r4);
			int xcorr2 = (x / cons) + centerx;
			int ycorr2 = (y / cons) + centery;
			mapx2.at<float>(i, j) = (float)xcorr2;
			mapy2.at<float>(i, j) = (float)ycorr2;
		}
	}
	while (true) {
		/// Load image
		stream1.read(src);
		//stream1.read(src);
		imshow("Imagen original", src);

		remap(src, dst, mapx, mapy, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
		remap(src, dst2, mapx2, mapy2, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
		imshow("Imagen cojin", dst);
		imshow("Imagen barrel", dst2);
		
		if (waitKey(30) >= 0) {
			break;
		}
	}
}

void efectoFanstasma() {
	VideoCapture stream1(0);
	Mat src;
	Mat src2;
	Mat src3;
	Mat src4;
	Mat src5;
	Mat src6;
	Mat dst;
	float alfa = 0.8;
	stream1.read(src);
	stream1.read(src);
	stream1.read(src2);
	stream1.read(src2);
	stream1.read(src3);
	stream1.read(src3);
	stream1.read(src4);
	stream1.read(src4);
	stream1.read(src5);
	stream1.read(src5);
	stream1.read(src6);
	stream1.read(src6);
	stream1.read(dst);
	stream1.read(dst);

	while (true) {
		stream1.read(src);
		stream1.read(src);
		dst = 0.9 * dst + 0.1*(0.9*src + 0.1* (src2 * 0.9 + 0.1* (src3 * 0.9 + 0.1*( src4 * 0.9 + 0.1 *(0.9 *  src5 + 0.1 * src6)))));
		imshow("Efecto fantasma", dst);
		src6 = src5.clone();
		src5 = src4.clone();
		src4 = src3.clone();
		src3 = src2.clone();
		src2 = src.clone();	
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
	cout << "Efecto alien: Presiona esc para continuar\n";
	efectoAlien();
	destroyAllWindows();
	cout << "Reducir colores: Presiona esc para continuar\n";
	reducirColores(25);
	destroyAllWindows();
	cout << "Distorsion: Presiona esc para continuar\n";
	distorsionRadial();
	destroyAllWindows();
	cout << "Efecto fantasma: Presiona esc para continuar\n";
	efectoFanstasma();
	destroyAllWindows();
	return 0;
}



