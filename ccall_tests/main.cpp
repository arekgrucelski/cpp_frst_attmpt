#include <iostream>
#include <opencv/cv.h>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "lib_tmp.h"

#include <chrono>
#include <thread>

using namespace cv;
using namespace std;

int main( int argc, char **argv )
{
  Mat image,gray,prev_gray;
  VideoCapture *cap = openVideo();
  std::vector<Point2f> points, prev_points;
  if (cap == NULL) return 0;
//
  captureFrame( &image,*cap );
  giveGray( &gray,&image );
  giveGray( &prev_gray,&image );
  std::vector<uchar> status;
//

  cv::namedWindow("Testing window",1);
  for (;;) {

    	captureFrame( &image,*cap );
	giveGray( &gray,&image );

	char c = (char)waitKey(10);
        if( c == 27 ) // ESC
            break;
        switch( c )
        {
        case 'r':
	    giveInitPoints( &gray,&points );
	    cv::imshow("Testing window",gray);
	    prev_points.resize(points.size());
            break;
        case 'c':
            break;
        case 'n':
            break;
        }
	if (points.size() > 5) opticalFlowEstimation( &prev_gray,&gray,&points,&prev_points,&status );
  	for ( unsigned int i=0; i<points.size(); i++ ) {
  	        cv::circle( image,points[i],3,cv::Scalar(230,0,0),-1,8 );
  	}


	cv::imshow("Testing window",image);

	copyFrame( &prev_gray,&gray );
	std::swap(prev_points,points);
  }

  return 1;
}
