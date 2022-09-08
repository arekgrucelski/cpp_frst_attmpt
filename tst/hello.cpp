#include <string>
#include <iostream>
#include <opencv/cv.h>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "jlcxx/jlcxx.hpp"

struct tmp
{
  int a;
  cv::VideoCapture* b;
};

//using namespace cv;

extern int lib_fnctn( void )
{
  std::cout << "\nThis function seems to work" << 1 << "\n"
    "OpenCV version " << CV_VERSION << std::endl;

  return 1;
}

int openVideo( tmp bb )
{
   bb.b = new cv::VideoCapture();
   bb.b->open(0);
   if ( !cap->isOpened() )
   {
     std::cout << "\nThe stream cannot be opened\n";
     std::cout << "No device?\n";
     return NULL;
   }
   return 1;
}


std::string greet()
{
   return "hello, world";
}

JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
  mod.method("greet", &greet);
  mod.method("lib_fnctn", &lib_fnctn);
  mod.method("openVideo", &openVideo);
}
