#include <opencv2/opencv.hpp>
#include <iostream>

//added 1/29
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
 
using namespace cv;
using namespace std;
 
int main(){

     //cpu mats
     Mat myCam, myCamHSV, red_mask;

     //gpu mats
     cuda::GpuMat gpu_myCam, gpu_myCamHSV, gpu_red_mask;

     // setting range for red
     Scalar red_lower = Scalar(160, 100, 20);
     Scalar red_upper = Scalar(180, 255, 255);

     // Creating the capture object for camera
     VideoCapture cap(0);
     
     while (true){
          // Capturing through camera
          // "myCam" is a single frame from the "cap" camera stream
          cap >> myCam;

          // myCam to HSV _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
          // cpu method
          //cvtColor(myCam, myCamHSV, COLOR_BGR2HSV);

          // gpu method
          gpu_myCam.upload(myCam); // loading myCam frame into VRAM
          cuda::cvtColor(gpu_myCam, gpu_myCamHSV, COLOR_BGR2HSV); // gpu cvtColor operation
          gpu_myCamHSV.download(myCamHSV); // dumping converted frame back into the cpu mat
     
          // making red mask
          inRange(myCamHSV, red_lower, red_upper, red_mask); // HSV -> red_mask
     
          Mat kernal = getStructuringElement(MORPH_RECT, Size(5, 5));
          dilate(red_mask, red_mask, kernal);
     
          // Finding contours of red object 
          vector<vector<Point> > contours_red;
          vector<Vec4i> hierarchy;
          findContours(red_mask, contours_red, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

          // everything black except for red
          // cpu method
          // Mat red;
          // bitwise_and(myCam, myCam, red, red_mask);

          // gpu method
          Mat red;
          cuda::GpuMat gpu_red;
          gpu_red_mask.upload(red_mask);
          cuda::bitwise_not(gpu_myCam, gpu_red, gpu_red_mask); // bitwise_and throws error, this is a workaround
          cuda::bitwise_not(gpu_red, gpu_red, gpu_red_mask); // https://github.com/opencv/opencv/issues/20698
          gpu_red.download(red);
          //gpu_red.release(); 
          gpu_red_mask.release(); // red_mask needs to be destroyed in VRAM or else it leaves residual masking data. uncomment and run to see what i mean

     
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