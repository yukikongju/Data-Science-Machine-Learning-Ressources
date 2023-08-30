#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define w 400
using namespace cv;

void MyEllipse(Mat img, double angle);


int main(void) {

    char img_window[] = "Drawing";
    Mat img = Mat::zeros(w, w, CV_8UC3);
    MyEllipse(img, 90);

    imshow(img_window, img);
    moveWindow(img_window, 0, 200);

    waitKey(0);
    return 0;
}


void MyEllipse(Mat img, double angle) {
    int thickness = 2;
    int lineType = 8;

    ellipse(img, Point(w/2, w/2), Size(w/4, w/16), angle, 0, 360, 
	    Scalar(255, 0, 0), thickness, lineType);
}

