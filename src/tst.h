//#ifdef CxxJULIA
#include "jlcxx/jlcxx.hpp"
//#endif 		 // CxxJULIA
#include<chrono>


#ifndef tst_h__
#define tst_h__

#define WIDTH 7680
#define HEIGHT 4320
#define KERNEL_SIZE 3

/*
 * \todo create time elapsed function 
 */
#ifndef myTime_struct__
#define myTime_struct__
struct myTime
{
//  extern std::chrono::high_resolution_clock::time_point 
    uint8_t strt;
//  extern std::chrono::high_resolution_clock::time_point 
    uint8_t stp;
};
#endif // myTime_struct__

extern void start_all( void );
//
extern void fillData( float**,float** );
extern void compareData( float**, float** );
extern void convolve( float** ,float** ,float** );
//
extern std::string greet();
extern int lib_fnctn( int );
extern void tic( myTime *);
extern void tac( float* ); //myTime *);

#endif 		// tst_h__
