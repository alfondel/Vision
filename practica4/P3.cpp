
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>

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

#define ip "http://192.168.1.135:8080/video?x.mjpeg"

bool camera = false;
bool calibr = false;
bool surf = true;


Mat show1;
Mat show2;
Mat show3;
VideoCapture capture;

//Calibration of the camera
void calibrate() {
	int numBoards = 0;
	int numCornersHor;
	int numCornersVer;
	cout << "Numero de columnas:" << endl;
	cin >> numCornersHor;
	cout << "Numero de filas:" << endl;
	cin >> numCornersVer;
	cout << "Cantidad de tableros" << endl;
	cin >> numBoards;
	int numSquares = numCornersHor * numCornersVer;
	Size board_sz = Size(numCornersHor, numCornersVer);



	vector<vector<Point3f>> object_points;
	vector<vector<Point2f>> image_points;
	vector<Point2f> corners;
	int successes = 0;
	Mat image;
	Mat gray_image;
	cout << "Presiona cualquier tecla para capturar" << endl;
	while (true) {

		capture.read(image);
		capture.read(image);
		GaussianBlur(image, image, Size(3, 3), 0, 0);
		resize(image, show1, Size(image.cols*0.4, image.rows *0.4));
		imshow("Imagen siguiente", show1);
		if (waitKey(30) >= 0) {
			break;
		}
	}
	cout << "calculando" << endl;

	vector< Point3f > obj;
	for (int i = 0; i < numCornersVer; i++)
		for (int j = 0; j < numCornersHor; j++)
			obj.push_back(Point3f((float)j * numCornersHor *numCornersHor, (float)i * numCornersHor, 0));
	while (successes < numBoards) {
		cvtColor(image, gray_image, CV_BGR2GRAY);
		GaussianBlur(gray_image, gray_image, Size(3, 3), 0, 0);
		bool found = findChessboardCorners(gray_image, board_sz, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
		if (found) {
			cout << "found!" << endl;
			cornerSubPix(gray_image, corners, Size(11, 11), Size(-1, 1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
			drawChessboardCorners(gray_image, board_sz, corners, found);

		}
		if (found) {
			image_points.push_back(corners);
			object_points.push_back(obj);
			cout << "Snap stored !" << endl;
			successes++;
			if (successes >= numBoards)
				break;
		}
		resize(image, show1, Size(image.cols*0.4, image.rows *0.4));
		imshow("win1", show1);
		resize(gray_image, show2, Size(gray_image.cols*0.4, gray_image.rows *0.4));
		imshow("win2", show2);
		cout << "Presiona cualquier tecla para capturar" << endl;
		while (true) {

			capture.read(image);
			capture.read(image);
			resize(image, show1, Size(image.cols*0.4, image.rows *0.4));
			imshow("Imagen siguiente", show1);
			if (waitKey(30) >= 0) {
				break;
			}
		}
		cout << "calculando" << endl;
		int key = waitKey(1);
		if (key == 27) return;

	}
	Mat intrinsic = Mat(3, 3, CV_32FC1);
	Mat distCoeffs;
	vector<Mat> rvecs;
	vector<Mat> tvecs;
	intrinsic.ptr<float>(0)[0] = 1;
	intrinsic.ptr<float>(1)[1] = 1;
	calibrateCamera(object_points, image_points, image.size(), intrinsic, distCoeffs, rvecs, tvecs);
	Mat imageUndistorted;
	while (1) {
		capture >> image;
		undistort(image, imageUndistorted, intrinsic, distCoeffs);

		resize(image, show1, Size(image.cols*0.4, image.rows *0.4));
		imshow("win1", show1);
		resize(imageUndistorted, show2, Size(imageUndistorted.cols*0.4, imageUndistorted.rows *0.4));
		imshow("win2", show2);
		waitKey(1);
	}
}





int main(int argc, char * argv[]) {

	if (camera) {
		capture.open(ip);
		if (!capture.isOpened())
		{
			cout << "Camera not found" << endl;
			getchar();
			return 0;
		}
	}
	else {
		capture = VideoCapture(0);
	}
	if (calibr) {
		calibrate();
	}

	Mat show;
	Mat showPano;
	bool terminar = false;
	Mat panorama;
	Mat captura;

	cout << " Pulsa cualquier boton en la imagen para tomar la captura" << endl;

	while (true) {

		capture.read(panorama);
		capture.read(panorama);
		//cvtColor(panorama, panorama, COLOR_BGR2GRAY);

		resize(panorama, show, Size(panorama.cols*0.4, panorama.rows *0.4));
		imshow("Imagen siguiente", show);
		if (waitKey(30) >= 0) {
			break;
		}
	}
	while (!terminar) {
		//panorama finalizado
		cout << "Pulsa cualquier boton en la imagen para tomar la captura" << endl;

		while (true) {

			capture.read(captura);
			capture.read(captura);
			//cvtColor(panorama, panorama, COLOR_BGR2GRAY);

			resize(captura, show, Size(captura.cols*0.4, captura.rows *0.4));
			imshow("Imagen siguiente", show);
			if (waitKey(30) >= 0) {
				break;
			}
		}

		// detecting keypoints
		vector<KeyPoint> keypoints1, keypoints2;
		if (surf) {
			SurfFeatureDetector detector(400);
			detector.detect(panorama, keypoints1);
			detector.detect(captura, keypoints2);
		}
		else {
			SiftFeatureDetector detector(400);
			detector.detect(panorama, keypoints1);
			detector.detect(captura, keypoints2);
		}



		Mat descriptors1, descriptors2;
		if (surf) {
			SurfDescriptorExtractor extractor;
			extractor.compute(panorama, keypoints1, descriptors1);
			extractor.compute(captura, keypoints2, descriptors2);
		}
		else {
			SiftDescriptorExtractor extractor;
			extractor.compute(panorama, keypoints1, descriptors1);
			extractor.compute(captura, keypoints2, descriptors2);
		}



		BFMatcher matcher(NORM_L2);
		vector<vector<DMatch>> matches;
		int k = 2;
		//se calculan los matches
		matcher.knnMatch(descriptors1, descriptors2, matches, k);

		float minDist = 0.25;
		vector<cv::DMatch> matchesCorrec;
		vector<DMatch> matches2;
		if (surf) {
			for (int i = 0; i < matches.size(); ++i) {
				// umbral para eliminar los puntos malos
				if (matches[i][0].distance < minDist) {
					matchesCorrec.push_back(matches[i][0]);
				}
				if (matches[i][1].distance < minDist) {
					matches2.push_back(matches[i][1]);
				}
			}
		}
		else {
			for (int i = 0; i < matches.size(); ++i) {
				matchesCorrec.push_back(matches[i][0]);
				matches2.push_back(matches[i][1]);
			}
		}
		//Homografia

		Mat img_matches1, img_matches2;
		drawMatches(panorama, keypoints1, captura, keypoints2, matchesCorrec, img_matches1);
		drawMatches(panorama, keypoints1, captura, keypoints2, matches2, img_matches2);
		resize(img_matches1, show1, Size(img_matches1.cols*0.4, img_matches1.rows *0.4));
		imshow("matches 1", show1);
		//resize(img_matches2, show2, Size(img_matches2.cols*0.4, img_matches2.rows *0.4));
		//imshow("matches 2", show2);

		std::vector<Point2f> puntosImagen;
		std::vector<Point2f> puntosPanorama;


		for (size_t i = 0; i < matchesCorrec.size(); i++)
		{
			//-- Get the keypoints from the good matches
			puntosImagen.push_back(keypoints1[matchesCorrec[i].queryIdx].pt);
			puntosPanorama.push_back(keypoints2[matchesCorrec[i].trainIdx].pt);
		}

		Mat h = findHomography(puntosPanorama, puntosImagen, CV_RANSAC);
		vector<Point2f> esquinasP, esquinasI, esquinasItrans(4);

		esquinasI.push_back(cvPoint(0, 0));
		esquinasI.push_back(cvPoint(captura.cols, 0));
		esquinasI.push_back(cvPoint(captura.cols, captura.rows));
		esquinasI.push_back(cvPoint(0, captura.rows));

		esquinasP.push_back(cvPoint(0, 0));
		esquinasP.push_back(cvPoint(panorama.cols, 0));
		esquinasP.push_back(cvPoint(panorama.cols, panorama.rows));
		esquinasP.push_back(cvPoint(0, panorama.rows));

		perspectiveTransform(esquinasI, esquinasItrans, h);
		int ancho, alto;
		Mat t = Mat::eye(3, 3, CV_64FC1);

		float xmin = 99999999999;
		float ymin = 99999999999;

		float xmax = -9999999999;
		float ymax = -9999999999;

		for (int j = 0; j < 4; j++) {
			if (esquinasItrans[j].x< xmin) {
				xmin = esquinasItrans[j].x;
			}
			if (esquinasItrans[j].x> xmax) {
				xmax = esquinasItrans[j].x;
			}
			if (esquinasItrans[j].y< ymin) {
				ymin = esquinasItrans[j].y;
			}
			if (esquinasItrans[j].y> ymax) {
				ymax = esquinasItrans[j].y;
			}
		}
		for (int j = 0; j < 4; j++) {
			if (esquinasP[j].x< xmin) {
				xmin = esquinasP[j].x;
			}
			if (esquinasP[j].x> xmax) {
				xmax = esquinasP[j].x;
			}
			if (esquinasP[j].y< ymin) {
				ymin = esquinasP[j].y;
			}
			if (esquinasP[j].y> ymax) {
				ymax = esquinasP[j].y;
			}
		}
		//caso de que sea necesario trasladar las imagenes
		if (xmin<0) {
			t.at<double>(0, 2) = -xmin;
		}
		if (ymin < 0) {
			t.at<double>(1, 2) = -ymin;
		}

		ancho = xmax - xmin;
		alto = ymax - ymin;
		//se calculan las dimensiones de la imagen final.
		Mat resultado = Mat(Size(ancho, alto), CV_32F);

		//se calcula la transformacion para la imagen a añadir
		warpPerspective(captura, resultado, t*h, resultado.size(), INTER_LINEAR, BORDER_TRANSPARENT);
		//se desplaza la imagen original en caso de x o y negativas
		warpPerspective(panorama, resultado, t, resultado.size(), INTER_LINEAR, BORDER_TRANSPARENT);
	
		//resize para evitar salirnos de la pantalla visible
		resize(resultado, showPano, Size(resultado.cols*0.2, resultado.rows *0.2));
		imshow("Panorama", showPano);

		cout << "Presiona s/n para aceptar o cancelar la adicion de esa imagen" << endl;
		char key;
		while (true) {

			key = waitKey(33);
			if (key == 's') {
				panorama = resultado.clone();
				break;
			}
			if (key == 'n') {
				resize(panorama, showPano, Size(resultado.cols*0.2, resultado.rows *0.2));
				imshow("Panorama", showPano);
				break;
			}
		}

	}

}