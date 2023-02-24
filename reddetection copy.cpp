#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <math.h>
#include <string.h>
 
using namespace cv;
using namespace std;

int thresh = 50, N = 5;

// ------------ new functions ------------
// these should eventually be moved into its own class object (for optimization purposes) and then into its own header file library thing (for formatting purposes)


// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle( Point pt1, Point pt2, Point pt0 ){
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
static void findSquares( const Mat& image, vector<vector<Point> >& squares ){
    squares.clear();

//  Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    //pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    //pyrUp(pyr, timg, image.size());


    // blur will enhance edge detection
    Mat timg(image);
    medianBlur(image, timg, 9);
    Mat gray0(timg.size(), CV_8U), gray;
    vector<vector<Point> > contours;
    vector<Point> approx;

    // find squares in every color plane of the image
    for( int c = 0; c < 3; c++ ){
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        for( int l = 0; l < N; l++ ){
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 ){
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 5, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else{
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;
            }

            // find contours and store them all as a list
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            // test each contour
            for( size_t i = 0; i < contours.size(); i++ ){
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if( approx.size() == 4 && fabs(contourArea(Mat(approx))) > 1000 && isContourConvex(Mat(approx)) ){
                    double maxCosine = 0;
                    for( int j = 2; j < 5; j++){
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if( maxCosine < 0.3 ){
                        squares.push_back(approx);
                    }
                }
            }
        }
    }
}

// the function draws all the squares in the image
static void drawSquares( Mat& image, const vector<vector<Point> >& squares ){
    for( size_t i = 0; i < squares.size(); i++ ){
        const Point* p = &squares[i][0];

        int n = (int)squares[i].size();
        //dont detect the border
        if (p-> x > 3 && p->y > 3)
          polylines(image, &p, &n, 1, true, Scalar(0,255,0), 3, LINE_AA);
    }
}

Point getCenterOfMat(Mat frame){ 
     // for any given frame, calculates the center of it. Should only be ran once then stored!
    return Point(frame.cols*0.5, frame.rows*0.5);
}

Point getErrorFromScreenCenter(Point frame_center, Point target){ 
     // dx & dy are calculated here in a Point object. Can be accessed with .x and .y functions
    return Point(target.x - frame_center.x, target.y - frame_center.y);
}

void drawCorrectionVector(Mat frame, Point frame_center, Point target, bool drawComponents = false){ // function for graphically notating information about the dx & dy
     //possible optimization - prereq the getErrorFromScreenCenter in a variable and reference it with a pointer.
     //                        in fact most things here can be dereferenced pointers instead of instantiated
    Point error = getErrorFromScreenCenter(frame_center, target);
    line(frame, target, frame_center, Scalar(255, 255, 255));
    if(drawComponents){
            line(frame, Point(frame_center.x + error.x, frame_center.y), getCenterOfMat(frame), Scalar(255, 0, 0));
            line(frame, Point(frame_center.x + error.x, frame_center.y + error.y), Point(frame_center.x + error.x, frame_center.y), Scalar(0, 255, 0));
    }
}

// -------------------------------------
 
int main(){
    Mat myCam, myCamHSV, red_mask;
    int morph_size = 2;
    Mat element = getStructuringElement(MORPH_RECT, Size(2 * morph_size + 1,2 * morph_size + 1), Point(morph_size, morph_size));

     // Creating the capture object for camera
    VideoCapture cap(0);

    Scalar red_lower = Scalar(160, 100, 20);
    Scalar red_upper = Scalar(180, 255, 255);
     
    while (true){
          // Capturing through camera
          cap >> myCam;

          // myCam to HSV
          cvtColor(myCam, myCamHSV, COLOR_BGR2HSV);
     
          // setting range for red
          inRange(myCamHSV, red_lower, red_upper, red_mask);
     
          // making red mask

          // everything black except for red
          Mat red;
          bitwise_and(myCam, myCam, red, red_mask);

          // Finding contours of red object
          vector<vector<Point> > contours_red;
          vector<Vec4i> hierarchy;
          vector<vector<Point> > squares;

          findSquares(red, squares);
          // Tracking red color and drawing contours
          for (int i = 0; i < squares.size(); i++){
               double area = contourArea(squares[i]);
               if (area > 1500){
                    Moments M = moments(squares[i]);
                    int cx, cy;

                    if (M.m00 != 0){
                         cx = (int) (M.m10 / M.m00);
                         cy = (int) (M.m01 / M.m00);
                         drawSquares(red, squares);
                         circle(red, Point(cx, cy), 7, Scalar(255, 255, 255), -1);
                         drawCorrectionVector(red, getCenterOfMat(red), Point(cx, cy), true);
                    }
               }
          }

          // Display the frame 
          imshow("Red", red); // show red only camera
     
          // Terminate the program if 'q' is pressed
          if (waitKey(1) == 27){
               break;
          }
     }
     cap.release();
     return 0;
     } 