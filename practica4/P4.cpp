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




int main(int argc, char * argv[]) {
	Mat img1 = imread("C:/Users/Alfonso/Downloads/imagenesT2/vagon1.pgm", CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2 = imread("C:/Users/Alfonso/Downloads/imagenesT2/vagon2.pgm", CV_LOAD_IMAGE_GRAYSCALE);
	// detecting keypoints
	SurfFeatureDetector detector(400);
	vector<KeyPoint> keypoints1, keypoints2;
	detector.detect(img1, keypoints1);
	detector.detect(img2, keypoints2);

	//drawKeypoints(img1, keypoints1, img1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	//drawKeypoints(img2, keypoints2, img2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	//imshow("Img1 ",img1);
	//imshow("Img2 ", img2);

	SurfDescriptorExtractor extractor;
	Mat descriptors1, descriptors2;
	extractor.compute(img1, keypoints1, descriptors1);
	extractor.compute(img2, keypoints2, descriptors2);


	BFMatcher matcher(NORM_L2);
	vector<vector<DMatch>> matches;
	int k = 2;
	matcher.knnMatch(descriptors1, descriptors2, matches, k);
	// drawing the results
	vector<DMatch> match1;
	vector<DMatch> match2;
    vector<Point2f> pts_src;
    vector<Point2f> pts_dst;
	for (int i = 0; i<matches.size(); i++)
	{
		match1.push_back(matches[i][0]);
		match2.push_back(matches[i][1]);
		cout << "Distancias punto " << i << " Distancia 1 " << matches[i][0].distance << " Distancia 2 " << matches[i][1].distance << endl;

        int idx1 = matches[i][0].trainIdx;
        int idx2 = matches[i][1].queryIdx;
        //TO DO: get x and y from matches..
        pts_src.push_back(Point2f(i * 2, i * 2));
        pts_dst.push_back(Point2f(i * 2, i * 2));
	}

	Mat img_matches1, img_matches2;
	drawMatches(img1, keypoints1, img2, keypoints2, match1, img_matches1);
	drawMatches(img1, keypoints1, img2, keypoints2, match2, img_matches2);
	imshow("matches 1", img_matches1);
	imshow("matches 2", img_matches2);
	

	//Homografia
 
    

    
    // pts_src and pts_dst are vectors of points in source 
    // and destination images. They are of type vector<Point2f>. 
    // We need at least 4 corresponding points. 

    Mat h = findHomography(pts_src, pts_dst);

    // The calculated homography can be used to warp 
    // the source image to destination. im_src and im_dst are
    // of type Mat. Size is the size (width,height) of im_dst. 
    Mat dst;
    warpPerspective(img1, dst, h, Size(700,700));
    imshow("img1 transform", dst);

    cv::SURF aa;
  

    waitKey(0);
}
