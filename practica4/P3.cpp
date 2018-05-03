
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

//Mat ampliar(Mat panorama,Mat warpcaptura) {

//}

int main(int argc, char * argv[]) {
    Mat show;
    bool terminar = false;
    Mat panorama;
    Mat captura;
    VideoCapture stream1(0);
    cout << " Pulsa cualquier boton en la imagen para tomar la captura" << endl;

    while (true) {

        stream1.read(panorama);
        cvtColor(panorama,panorama,COLOR_BGR2GRAY);
        imshow("Imagen siguiente", panorama);
        if (waitKey(30) >= 0) {
            break;
        }
    }
    while (!terminar){
        //panorama finalizado
        cout << "Pulsa cualquier boton en la imagen para tomar la captura" << endl;
      
        while(true){

            stream1.read(captura);
            cvtColor(captura, captura, COLOR_BGR2GRAY);
            imshow("Imagen siguiente", captura);
            if (waitKey(30) >= 0) {
                break;
           }
        } 
    
        // detecting keypoints
      
        SurfFeatureDetector detector(400);
        vector<KeyPoint> keypoints1, keypoints2;
        detector.detect(panorama, keypoints1);
        detector.detect(captura, keypoints2);

        SurfDescriptorExtractor extractor;
        Mat descriptors1, descriptors2;
        extractor.compute(panorama, keypoints1, descriptors1);
        extractor.compute(captura, keypoints2, descriptors2);


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
        drawMatches(panorama, keypoints1, captura, keypoints2, matchesCorrec, img_matches1);
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
        vector<Point2f> esquinasP(4), esquinasI(4),esquinasItrans(4);

        esquinasI[0] = cvPoint(0, 0);
        esquinasI[1] = cvPoint(captura.cols, 0);
        esquinasI[2] = cvPoint(captura.cols, captura.rows);
        esquinasI[3] = cvPoint(0, captura.rows);

        esquinasP[0] = cvPoint(0, 0);
        esquinasP[1] = cvPoint(panorama.cols, 0);
        esquinasP[2] = cvPoint(panorama.cols, panorama.rows);
        esquinasP[3] = cvPoint(0, panorama.rows);

        perspectiveTransform(esquinasI, esquinasItrans, h);

        float minX=99999999999;
        float minY=99999999999;

        float maxX=-9999999999;
        float maxY=-9999999999;
        
        for (int j = 0; j < 4; j++) {
            if (esquinasItrans[j].x< minX) {
                minX = esquinasItrans[j].x;
            }
            if (esquinasItrans[j].x> maxX) {
                maxX = esquinasItrans[j].x;
            }
            if (esquinasItrans[j].y< minY) {
                minY = esquinasItrans[j].y;
            }
            if (esquinasItrans[j].y> maxY) {
                maxY = esquinasItrans[j].y;
            }
        }
        for (int j = 0; j < 4; j++) {
            if (esquinasP[j].x< minX) {
                minX = esquinasP[j].x;
            }
            if (esquinasP[j].x> maxX) {
                maxX = esquinasP[j].x;
            }
            if (esquinasP[j].y< minY) {
                minY = esquinasP[j].y;
            }
            if (esquinasP[j].y> maxY) {
                maxY = esquinasP[j].y;
            }
        }
        int width = maxX - minX;
        int height = maxY - minY;
        
        Mat t = Mat::eye(3, 3, CV_64FC1);
        if (minX<0){
            t.at<double>(0, 2) = -minX;
        }
        if (minY < 0) {
            t.at<double>(1, 2) = -minY;
        }
        Mat result = Mat(Size(width, height), CV_32F);
        warpPerspective(panorama, result, t, result.size(), INTER_LINEAR, BORDER_TRANSPARENT);
        warpPerspective(captura, result, t*h, result.size(), INTER_LINEAR, BORDER_TRANSPARENT);

        resize(result, show, Size(result.cols*0.7, result.rows *0.7));

        
      
        imshow("Panorama", show);
        panorama = result.clone();
       
       
    }
    
}
