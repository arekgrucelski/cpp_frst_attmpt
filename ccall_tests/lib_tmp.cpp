#include <iostream>
#include <opencv/cv.h>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

void lib_fnctn( cv::Mat image,int dana_int ) 
{
  std::cout << "\nThis function seems to work" << dana_int << "\n"
    "OpenCV version " << CV_VERSION << std::endl;

  return;
}

cv::VideoCapture* openVideo( void )
{
   cv::VideoCapture *cap = new cv::VideoCapture();
   cap->open(0);
   if ( !cap->isOpened() )
   {
     std::cout << "\nThe stream cannot be opened\n";
     std::cout << "No device?\n";
     return NULL;
   }
   return cap;
}

void captureFrame( cv::Mat *frame,cv::VideoCapture cap )
{
#ifdef _DEBUG__
  std::cout << "W = " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
  std::cout << "H = " << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
#endif // _DEBUG__

  cap.read( *frame );
  return;
}

void giveGray( cv::Mat *gray, cv::Mat *rgb )
{
  cv::cvtColor( *rgb,*gray,cv::COLOR_BGR2GRAY );
}

#define MAX_COUNT 130
void giveInitPoints( cv::Mat *gray,std::vector<Point2f> *points  ) //,std::vector<Point2f>
{
  cv::TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
  cv::Size subPixWinSize(10,10); //, winSize(31,31);
  cv::goodFeaturesToTrack(*gray, *points, MAX_COUNT, 0.01, 10, cv::Mat(), 3, 0, 0.04);
  cv::cornerSubPix(*gray, *points, subPixWinSize, cv::Size(-1,-1), termcrit);
//
#ifdef _DEBUG__
  std::cout << "r key pressed:" << std::endl;
  std::cout << "\tnumber of points " << points->size() << std::endl;
#endif	// _DEBUG__
}

void copyFrame( cv::Mat *prv_img,cv::Mat *img )
{
  (*prv_img).copyTo(*img);
}

void opticalFlowEstimation( 
    cv::Mat* img, cv::Mat * prev_img,
    std::vector<cv::Point2f> *points,std::vector<cv::Point2f> *prev_points, std::vector<uchar> *status )
{
  std::vector<float> err;
  cv::TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.01);

  calcOpticalFlowPyrLK( *prev_img, *img, *points, *prev_points, 
	*status, err, cvSize(31,31), 3, termcrit, 0, 1e-3); //, winSize, 3, termcrit, 0, 0.001);
  calcOpticalFlowPyrLK( *img, *prev_img, *prev_points, *points, 
	*status, err, cvSize(31,31), 3, termcrit, 0, 1e-3); //, winSize, 3, termcrit, 0, 0.001);
/*  for (i=0;i<points.size(),i++) {
    points
  }*/
    
#ifdef _DEBUG__
  std::cout << "flow check" << std::endl;
  std::cout << "\tnumber of points " << points->size() << std::endl;
#endif	// _DEBUG__
}



