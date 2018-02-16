// ConsoleApplication1.cpp: define el punto de entrada de la aplicación de consola.
//

#include "stdafx.h" 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std; 

int main()
{
	VideoCapture stream1(0);
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
		imshow("cam1", cameraFrame);
		cameraFrame = cameraFrame * a;
		cameraFrame = cameraFrame + b;
		imshow("cam2", cameraFrame);
		if (waitKey(30)>=0) {
			break;
		}
	}

	stream1.release();
	destroyAllWindows();
	return 0;
}

