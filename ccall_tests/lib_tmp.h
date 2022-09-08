#ifndef lib_tmp__
#define lib_tmp__
#include <opencv/cv.h>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>


extern void lib_fnctn( int );
extern cv::VideoCapture* openVideo( void ); 
extern void captureFrame( cv::Mat *,cv::VideoCapture );
extern void giveGray( cv::Mat *, cv::Mat *);
extern void giveInitPoints( cv::Mat *,std::vector<cv::Point2f> * );
extern void copyFrame( cv::Mat* , cv::Mat * );
extern void opticalFlowEstimation( 
    cv::Mat* , cv::Mat *,
    std::vector<cv::Point2f> *,std::vector<cv::Point2f> *,
    std::vector<uchar> * );

#endif
