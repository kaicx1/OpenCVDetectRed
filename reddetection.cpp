#include <opencv2/opencv.hpp>
#include <iostream>
 
using namespace cv;
using namespace std;
 
int main(){
     Mat myCam, myCamHSV, red_mask;
     // Creating the capture object for camera
     VideoCapture cap(0);
     
     while (true){
          // Capturing through camera
          cap >> myCam;

          // myCam to HSV
          cvtColor(myCam, myCamHSV, COLOR_BGR2HSV);
     
          // setting range for red
          Scalar red_lower = Scalar(160, 100, 20);
          Scalar red_upper = Scalar(180, 255, 255);
     
          // making red mask
          inRange(myCamHSV, red_lower, red_upper, red_mask);
     
          Mat kernal = getStructuringElement(MORPH_RECT, Size(5, 5));
          dilate(red_mask, red_mask, kernal);
     
          // Finding contours of red object 
          vector<vector<Point> > contours_red;
          vector<Vec4i> hierarchy;
          findContours(red_mask, contours_red, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

          // everything black except for red
          Mat red;
          bitwise_and(myCam, myCam, red, red_mask);
     
          // Tracking red color and drawing contours
          for (int i = 0; i < contours_red.size(); i++){
               double area = contourArea(contours_red[i]);
               if (area > 1500){
                    // Rectangles for red objects 
                    Rect rect = boundingRect(contours_red[i]);
                    rectangle(red, boundingRect(contours_red[i]), Scalar(0, 0, 255), 2);

                    Moments M = moments(contours_red[i]);
                    int cx, cy;

                    if (M.m00 != 0){
                         cx = (int) (M.m10 / M.m00);
                         cy = (int) (M.m01 / M.m00);
                         drawContours(red, contours_red, i, Scalar(0, 255, 0), 2);
                         circle(red, Point(cx, cy), 7, Scalar(0, 0, 255), -1);
                         cout << "x: " << cx << ", y: " << cy << endl;
                    }
               }
          }
          // Display the frame 
          imshow("Original", myCam);
          imshow("Red", red); // show red only camera
     
          // Terminate the program if 'q' is pressed
          if (waitKey(1) == 27){
               break;
          }
     }
     cap.release();
     return 0;
     } 