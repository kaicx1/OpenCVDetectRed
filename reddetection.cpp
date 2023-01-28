#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;
int main( int argc, const char** argv ){
     VideoCapture cap(0);
     Mat myCam, myCam_hsv, redMask;
     if(!cap.isOpened()){ // If no camera return -1 
          return -1;
     }
     while(1){
          cap >> myCam; // Open Camera
          cvtColor(myCam, myCam_hsv, COLOR_BGR2HSV); // camera to HSV
          Vec3b L_limit(160,100,20); // Setting lower limit
          Vec3b U_limit(179,255,255); // Setting upper limit
          inRange(myCam_hsv, L_limit, U_limit, redMask);
          Mat red;
          bitwise_and(myCam, myCam, red, redMask);
          imshow("Original", myCam); //  show original camera 
          imshow("Red", red); // show red only camera

          if(waitKey(1) == 27){
               break;
          }
    }
    return 0;
}