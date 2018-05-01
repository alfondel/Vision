#define _USE_MATH_DEFINES
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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

Mat panorama; 


int main(int argc, char * argv[]) {
	vector<Mat> imagenes; 
	Mat img1 = imread("C:/Users/Alfonso/Downloads/foto1_opt.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2 = imread("C:/Users/Alfonso/Downloads/foto2_opt.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat img3 = imread("C:/Users/Alfonso/Downloads/foto3_opt.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	imagenes.push_back(img1);
	imagenes.push_back(img2);
	imagenes.push_back(img3);
	imshow("imagen 1", img1);
	imshow("imagen 2", img2);
	imshow("imagen 3", img3);
	waitKey(0);
	cout << "Pulsa D para sentido derecha o I para sentido izquierda." << endl;
	char tecla;
	int numImgs = 3;
	cin >> tecla;
	int offsetx;
	//Ampliar imagen para añadir luego las demas 
	if (tecla==68 || tecla == 100) {
		offsetx = 0;
	}
	else if(tecla == 105 || tecla == 73) {
		offsetx = img1.cols*(numImgs-1);
	}
	int offsety = 0;
	
	Mat trans_mat = (Mat_<double>(2, 3) << 1, 0, offsetx, 0, 1, offsety);
	warpAffine(img1, img1, trans_mat, Size(img1.cols * numImgs,img1.rows));
	imshow("Imagen BASE", img1);
	waitKey(0);
	// detecting keypoints
	for (int i = 1; i < imagenes.size();i++) {
		img2 = imagenes[i];
		imshow("img 2", img2);
		waitKey(0);
		SurfFeatureDetector detector(400);
		vector<KeyPoint> keypoints1, keypoints2;
		detector.detect(img1, keypoints1);
		detector.detect(img2, keypoints2);

		SurfDescriptorExtractor extractor;
		Mat descriptors1, descriptors2;
		extractor.compute(img1, keypoints1, descriptors1);
		extractor.compute(img2, keypoints2, descriptors2);


		BFMatcher matcher(NORM_L2);
		vector<vector<DMatch>> matches;
		int k = 2;
		matcher.knnMatch(descriptors1, descriptors2, matches, k);

		float minDist = 0.25;
		vector<cv::DMatch> matchesCorrec;
		for (int i = 0; i < matches.size(); ++i) {
			// umbral para cargarte los puntos malos
			if (matches[i][0].distance < minDist) {
				matchesCorrec.push_back(matches[i][0]);
			}
		}
		//Homografia

		Mat img_matches1, img_matches2;
		drawMatches(img1, keypoints1, img2, keypoints2, matchesCorrec, img_matches1);
		//drawMatches(img1, keypoints1, img2, keypoints2, match2, img_matches2);
		//imshow("matches 1", img_matches1);
		//imshow("matches 2", img_matches2);

		std::vector<Point2f> puntosImagen;
		std::vector<Point2f> puntosPanorama;
		for (size_t i = 0; i < matchesCorrec.size(); i++)
		{
			//-- Get the keypoints from the good matches
			puntosImagen.push_back(keypoints1[matchesCorrec[i].queryIdx].pt);
			puntosPanorama.push_back(keypoints2[matchesCorrec[i].trainIdx].pt);
		}

		// pts_src and pts_dst are vectors of points in source 
		// and destination images. They are of type vector<Point2f>. 
		// We need at least 4 corresponding points. 

		Mat h = findHomography(puntosPanorama, puntosImagen, CV_RANSAC);

		// The calculated homography can be used to warp 
		// the source image to destination. im_src and im_dst are
		// of type Mat. Size is the size (width,height) of im_dst.
		Mat dst;
		//warpPerspective(img1, dst, h, Size(img1.cols, img1.rows));
		//imshow("img1 transform", dst);
		warpPerspective(img2, dst, h, img1.size());

		//imshow("img2 transform", dst);

		for (int i = 0; i < img1.cols; i++) {
			for (int j = 0; j < img1.rows; j++) {
				uchar color_im1 = img1.at<uchar>(Point(i, j));
				uchar color_im2 = dst.at<uchar>(Point(i, j));
				if (norm(color_im1) == 0) {
					img1.at<uchar>(Point(i, j)) = color_im2;
				}
			}
		}
	}
	//panorama finalizado
	imshow("Panorama", img1);

	waitKey(0);
}