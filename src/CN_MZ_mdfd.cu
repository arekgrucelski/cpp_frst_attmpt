#include <iomanip>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <math.h>
#include <cuda.h>
#include <omp.h>
#include <random>
#include <stdio.h>
#include <chrono>

#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <sys/resource.h>

#include <cudnn.h>
#include "cublas_v2.h"

#define IMAGE_N 1
#define IMAGE_C 3
#define IMAGE_H 480
#define IMAGE_W 640

#define FatalError(s) {                                                \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;\
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(EXIT_FAILURE);                                                \
}

#define checkCUDNN(status) {                                           \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure\nError: " << cudnnGetErrorString(status); \
      FatalError(_error.str());                                        \
    }                                                                  \
}

#define checkCudaErrors(status) {                                      \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure\nError: " << cudaGetErrorString(status); \
      FatalError(_error.str());                                        \
    }                                                                  \
}




__global__ void dev_const(float *px, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
 
  curandState state;
  curand_init(clock64(), tid, 0, &state);

  if (tid < n)
    px[tid] = curand_uniform(&state);
}

cudaError_t generate_kernel_gpu(float *filt_data, int filt_w, int filt_h, int filt_c, int filt_k){

  int blk_size = 256;

  dim3 dimBlock(blk_size);

  int n = filt_k*filt_c*filt_h*filt_w;
  dim3 dimGrid_f = dim3((n + dimBlock.x -1)/dimBlock.x);
  dev_const<<<dimGrid_f, dimBlock>>>(filt_data, n);

  return cudaGetLastError();
}

void generate_data(float *array, int size, bool is_output){
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distr(-1.0f,1.0f);

    if(is_output){
        for(int i=0; i<size;i++){
		    array[i]=0;
	    } 
    }
	for(int i=0; i<size;i++){
		array[i]=distr(gen);
	}
}

void print_cpu(float *array, int n, int channels, int height, int width){
  int size = n*channels*height*width;
  for(int i=0;i<size;i++){       
		printf("%f ", array[i]);
        if(((i+1) % (channels*width*height) == 0)){
            printf("\n \n \n");
        }else if(((i+1) % (width*height) == 0)){
            printf("\n \n");
        }else if(((i+1) % (width)) == 0 || ((i+1) % (height)) == 0){
			printf("\n");
		}
  }
}

void print(const float *data, int n, int c, int h, int w) {
  std::vector<float> buffer(1 << 20);
  checkCudaErrors(cudaMemcpy(
        buffer.data(), data,
        n * c * h * w * sizeof(float),
        cudaMemcpyDeviceToHost));
  int a = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      std::cout << "n=" << i << ", c=" << j << ":" << std::endl;
      for (int k = 0; k < h; ++k) {
        for (int l = 0; l < w; ++l) {
          std::cout << std::setw(12) << std::right << buffer[a];
          ++a;
        }
        std::cout << std::endl << std::endl;
      }
    }
  }
  std::cout << std::endl;
}

void binRead(float *outputArray, const std::string& fileName, const unsigned int size){
  std::ifstream inputData;
  inputData.open(fileName);
  if(inputData){
    inputData.read(reinterpret_cast<char *>(outputArray), sizeof(float) * size);
    inputData.close();
  }
  else{
    std::cout << "Cannot open file" << std::endl;
  }
}

void binWrite(float *inputArray, const std::string& fileName, const unsigned int size){
  std::ofstream outputData;
  outputData.open(fileName);
  if(outputData){
    outputData.write(reinterpret_cast<const char *>(inputArray), sizeof(float) * size);
    outputData.close();
  }
  else{
    std::cout << "Cannot open file" << std::endl;
  }
}

void normalize(float* norm_array, int n, int c, int h, int w){
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);
	float *single_vector = NULL;
	checkCudaErrors(cudaMalloc(&single_vector, h*w*sizeof(float)));
	float *result = new float[1];
	cublasScopy(cublasHandle, h*w, norm_array, 1, single_vector, 1);
	cublasSasum(cublasHandle, h*w, single_vector, 1, result);
	float alpha[] = {1/result[0]};
	cublasSscal(cublasHandle, n*c*h*w, alpha, norm_array, 1);
	checkCudaErrors(cudaFree(single_vector));
	delete [] result;
}

/*
    Blck AG mdfctn of MZ code
*/
struct Layer_ag
{
    std::string name;
    cudnnConvolutionFwdAlgo_t alg;
    int inputs;
    int outputs;
    int kernel_dim;
    int pad_h, pad_w, str_h, str_w;
    float *data_d, *data_h;

    /*initiall constructor*/
    Layer_ag() : 
                    name("blad"),
                    alg((cudnnConvolutionFwdAlgo_t)0),
                    inputs(0),
                    outputs(0),
                    kernel_dim(0),
                    pad_h(0),
                    pad_w(0),
                    str_h(0),
                    str_w(0),
                    data_d(NULL),
                    data_h(NULL)
                    {
                        std::cout << "possible error occured as empty layer was created" << std::endl;
                    }
    Layer_ag(std::string _name, 
            cudnnConvolutionFwdAlgo_t _algo,
            int _inputs, 
            int _outputs, 
            int _kernel_dim,
            int _pad_h, 
            int _pad_w, 
            int _str_h, 
            int _str_w ) : 
                    name(_name), 
                    alg(_algo),
                    inputs(_inputs), 
                    outputs(_outputs), 
                    kernel_dim(_kernel_dim),
                    pad_h(_pad_h), 
                    pad_w(_pad_w), 
                    str_h(_str_h), 
                    str_w(_str_w)
                    {  
                        bool TEST = false;
                        data_h = new float [outputs*inputs*kernel_dim*kernel_dim];
                        checkCudaErrors(cudaMalloc(&data_d, outputs*inputs*kernel_dim*kernel_dim*sizeof(float)));
                        std::string filename = "src/filters/" + name + ".bin";
                        if(!TEST){
                            binRead(data_h, filename, outputs*inputs*kernel_dim*kernel_dim);
                        }
                        else{
                            generate_data(data_h, outputs*inputs*kernel_dim*kernel_dim, false);
                            binWrite(data_h, filename, outputs*inputs*kernel_dim*kernel_dim);
                        }
                        checkCudaErrors(cudaMemcpy(data_d, data_h, outputs*inputs*kernel_dim*kernel_dim*sizeof(float), cudaMemcpyHostToDevice));
                    }
    ~Layer_ag()
    {
        if (data_h != NULL) delete [] data_h;
        if (data_d != NULL) checkCudaErrors( cudaFree(data_d) );
    }

};
/*
end AG mdfctn
*/

template <class value_type>
struct Layer_t
{
    int inputs;
    int outputs;
    int kernel_dim;
    int pad_h, pad_w, str_h, str_w;
    value_type *data_h, *data_d;
    std::string name;
    std::string defaultName = "layer";
    cudnnConvolutionFwdAlgo_t algo;
    Layer_t() : name(defaultName), algo((cudnnConvolutionFwdAlgo_t)0),
                data_h(NULL), data_d(NULL), 
                inputs(0), outputs(0), kernel_dim(0),
                pad_h(0), pad_w(0), str_h(0), str_w(0){};
    Layer_t(std::string _name, cudnnConvolutionFwdAlgo_t _algo,
            int _inputs, int _outputs, int _kernel_dim,
            int _pad_h, int _pad_w, int _str_h, int _str_w)
                  : name(_name), algo(_algo),
                  inputs(_inputs), outputs(_outputs), kernel_dim(_kernel_dim),
                  pad_h(_pad_h), pad_w(_pad_w), str_h(_str_h), str_w(_str_w)
    {
      bool TEST = false;
      data_h = new value_type [outputs*inputs*kernel_dim*kernel_dim];
      checkCudaErrors(cudaMalloc(&data_d, outputs*inputs*kernel_dim*kernel_dim*sizeof(value_type)));
      std::string filename = "src/filters/" + name + ".bin";
      if(!TEST){
        binRead(data_h, filename, outputs*inputs*kernel_dim*kernel_dim);
      }
      else{
        generate_data(data_h, outputs*inputs*kernel_dim*kernel_dim, false);
        binWrite(data_h, filename, outputs*inputs*kernel_dim*kernel_dim);
      }
      checkCudaErrors(cudaMemcpy(data_d, data_h, outputs*inputs*kernel_dim*kernel_dim*sizeof(float), cudaMemcpyHostToDevice));
    }
    ~Layer_t()
    {
        if (data_h != NULL) delete [] data_h;
        //if (data_d != NULL) checkCudaErrors( cudaFree(data_d) );
    }

};

#define ND_TENSOR_DESCRIPTOR
void setTensorDesc(cudnnTensorDescriptor_t& tensorDesc, 
                    cudnnTensorFormat_t& tensorFormat,
                    cudnnDataType_t& dataType,
                    int n,
                    int c,
                    int h,
                    int w)
{
#if SIMPLE_TENSOR_DESCRIPTOR
    checkCUDNN( cudnnSetTensor4dDescriptor(tensorDesc,
                                            tensorFormat,
                                            dataType,
                                            n, c,
                                            h,
                                            w ) );
#elif defined(ND_TENSOR_DESCRIPTOR)
    const int nDims = 4;
    int dimA[nDims] = {n,c,h,w};
    int strideA[nDims] = {c*h*w, h*w, w, 1};
    checkCUDNN( cudnnSetTensorNdDescriptor(tensorDesc,
                                            dataType,
                                            4,
                                            dimA,
                                            strideA ) ); 
#else
    checkCUDNN( cudnnSetTensor4dDescriptorEx(tensorDesc,
                                            dataType,
                                            n, c,
                                            h, w,
                                            c*h*w, h*w, w, 1) );
#endif
}

template <class value_type>
class network_t
{
    int convAlgorithm;
    cudnnDataType_t dataType;
    cudnnTensorFormat_t tensorFormat;
    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    void createHandles()
    {
        cudnnCreate(&cudnnHandle);
        cudnnCreateTensorDescriptor(&srcTensorDesc);
        cudnnCreateTensorDescriptor(&dstTensorDesc);
        cudnnCreateFilterDescriptor(&filterDesc);
        cudnnCreateConvolutionDescriptor(&convDesc);

    }
    void destroyHandles()
    {
        cudnnDestroyConvolutionDescriptor(convDesc);
        cudnnDestroyFilterDescriptor(filterDesc);
        cudnnDestroyTensorDescriptor(srcTensorDesc);
        cudnnDestroyTensorDescriptor(dstTensorDesc);
        cudnnDestroy(cudnnHandle);
    }
  public:
    network_t()
    {
        convAlgorithm = -1;
        dataType = CUDNN_DATA_FLOAT;
        tensorFormat = CUDNN_TENSOR_NCHW;
        createHandles();    
    };
    ~network_t()
    {
        destroyHandles();
    }
    void resize(int size, value_type **data)
    {
        if (*data != NULL)
        {
            checkCudaErrors( cudaFree(*data) );
        }
        cudaMalloc(data, size*sizeof(value_type));
    }
    void setConvolutionAlgorithm(const cudnnConvolutionFwdAlgo_t& algo)
    {
        convAlgorithm = (int) algo;
    }


/*
    Blck AG mdfctn of MZ code
*/
    void convoluteForward(const Layer_ag &conv,
                          int& n, int& c, int& h, int& w,
                          value_type* srcData, value_type** dstData)
    {
        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);

        const int tensorDims = 4;
        int tensorOuputDimA[tensorDims] = {n,c,h,w};
        const int filterDimA[tensorDims] = {conv.outputs, conv.inputs, 
                                        conv.kernel_dim, conv.kernel_dim};
                                       
        cudnnSetFilterNdDescriptor(filterDesc,
                                              dataType,
                                              CUDNN_TENSOR_NCHW,
                                              tensorDims,
                                              filterDimA);
 
        const int convDims = 2;
        int padA[convDims] = {conv.pad_h, conv.pad_w};
        int filterStrideA[convDims] = {conv.str_h, conv.str_w};
        int upscaleA[convDims] = {1,1};
        cudnnSetConvolutionNdDescriptor(convDesc,
                                                    convDims,
                                                    padA,
                                                    filterStrideA,
                                                    upscaleA,
                                                    CUDNN_CROSS_CORRELATION,
                                                    CUDNN_DATA_FLOAT);
        // find dimension of convolution output
        cudnnGetConvolutionNdForwardOutputDim(convDesc,
                                                srcTensorDesc,
                                                filterDesc,
                                                tensorDims,
                                                tensorOuputDimA);
        n = tensorOuputDimA[0]; c = tensorOuputDimA[1];
        h = tensorOuputDimA[2]; w = tensorOuputDimA[3];

        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);
        
        resize(n*c*h*w, dstData);
        size_t sizeInBytes=0;
        void* workSpace=NULL;
        cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                srcTensorDesc,
                                                filterDesc,
                                                convDesc,
                                                dstTensorDesc,
                                                (cudnnConvolutionFwdAlgo_t)conv.alg,
                                                &sizeInBytes);
        
        if (sizeInBytes!=0)
        {
          cudaMalloc(&workSpace,sizeInBytes);
        }
        
        float alpha = 1.0f;
        float beta  = 0.0f;  
        cudnnConvolutionForward(cudnnHandle,
                                              &alpha,
                                              srcTensorDesc,
                                              srcData,
                                              filterDesc,
                                              conv.data_d,
                                              convDesc,
                                              (cudnnConvolutionFwdAlgo_t)conv.alg,
                                              workSpace,
                                              sizeInBytes,
                                              &beta,
                                              dstTensorDesc,
                                              *dstData);                                    
        if (sizeInBytes!=0)
        {
          cudaFree(workSpace);
        }
    }
/*
end AG mdfctn
*/
    
    void convoluteForward(const Layer_t<value_type>& conv,
                          int& n, int& c, int& h, int& w,
                          value_type* srcData, value_type** dstData)
    {
        //cudnnConvolutionFwdAlgo_t algo;

        //printf("input\n");
        //print(srcData,n,c,h,w);
        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);


        const int tensorDims = 4;
        int tensorOuputDimA[tensorDims] = {n,c,h,w};
        const int filterDimA[tensorDims] = {conv.outputs, conv.inputs, 
                                        conv.kernel_dim, conv.kernel_dim};
                                       
        cudnnSetFilterNdDescriptor(filterDesc,
                                              dataType,
                                              CUDNN_TENSOR_NCHW,
                                              tensorDims,
                                              filterDimA);
 
        const int convDims = 2;
        int padA[convDims] = {conv.pad_h, conv.pad_w};
        int filterStrideA[convDims] = {conv.str_h, conv.str_w};
        int upscaleA[convDims] = {1,1};
        cudnnSetConvolutionNdDescriptor(convDesc,
                                                    convDims,
                                                    padA,
                                                    filterStrideA,
                                                    upscaleA,
                                                    CUDNN_CROSS_CORRELATION,
                                                    CUDNN_DATA_FLOAT);
        // find dimension of convolution output
        cudnnGetConvolutionNdForwardOutputDim(convDesc,
                                                srcTensorDesc,
                                                filterDesc,
                                                tensorDims,
                                                tensorOuputDimA);
        n = tensorOuputDimA[0]; c = tensorOuputDimA[1];
        h = tensorOuputDimA[2]; w = tensorOuputDimA[3];

        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);
        
        /*if (convAlgorithm < 0)
        {
            // New way of finding the fastest config
            // Setup for findFastest call
            //std::cout << "Testing cudnnFindConvolutionForwardAlgorithm ...\n";
            int requestedAlgoCount = 5; 
            int returnedAlgoCount[1];
            cudnnConvolutionFwdAlgoPerf_t *results = (cudnnConvolutionFwdAlgoPerf_t*)malloc(sizeof(cudnnConvolutionFwdAlgoPerf_t)*requestedAlgoCount);
            cudnnFindConvolutionForwardAlgorithm( cudnnHandle, 
                                                     srcTensorDesc,
                                                     filterDesc,
                                                     convDesc,
                                                     dstTensorDesc,
                                                     requestedAlgoCount,
                                                     returnedAlgoCount,
                                                     results
                                                   );
        for(int algoIndex = 0; algoIndex < *returnedAlgoCount; ++algoIndex){
            printf("^^^^ %s for Algo %d: %f time requiring %llu memory\n", cudnnGetErrorString(results[algoIndex].status), results[algoIndex].algo, results[algoIndex].time, (unsigned long long)results[algoIndex].memory);
        }
            algo = results[0].algo;
            free(results);
        }
        else
        {
            algo = (cudnnConvolutionFwdAlgo_t)convAlgorithm;
        }*/
        resize(n*c*h*w, dstData);
        size_t sizeInBytes=0;
        void* workSpace=NULL;
        cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                srcTensorDesc,
                                                filterDesc,
                                                convDesc,
                                                dstTensorDesc,
                                                (cudnnConvolutionFwdAlgo_t)conv.algo,
                                                &sizeInBytes);
        
        if (sizeInBytes!=0)
        {
          cudaMalloc(&workSpace,sizeInBytes);
        }
        
        float alpha = 1.0f;
        float beta  = 0.0f;  
        cudnnConvolutionForward(cudnnHandle,
                                              &alpha,
                                              srcTensorDesc,
                                              srcData,
                                              filterDesc,
                                              conv.data_d,
                                              convDesc,
                                              (cudnnConvolutionFwdAlgo_t)conv.algo,
                                              workSpace,
                                              sizeInBytes,
                                              &beta,
                                              dstTensorDesc,
                                              *dstData);                                    
        // printf("Kernel\n");
        // print(conv.data_d, conv.outputs, conv.inputs, 3, 3);
        //printf("Output\n");
        //print(*dstData, 1, c, h, w);
        if (sizeInBytes!=0)
        {
          cudaFree(workSpace);
        }
    }

/*
    Blck AG mdfctn of MZ code
*/
    void convNext_backbone(Layer_ag *conv, Layer_ag *conv_block)
    {
        int n,c,h,w;
        int image_size = IMAGE_N*IMAGE_C*IMAGE_H*IMAGE_W;
        float *srcData = NULL, *dstData = NULL;
        float *imgData_h = new float[image_size*sizeof(float)];
        bool TEST = false;

        if(!TEST){
          binRead(imgData_h, "src/input.bin", image_size*sizeof(float));
        }
        else{
          generate_data(imgData_h, image_size, false);
          binWrite(imgData_h, "src/input.bin", image_size);
        }
        
        std::cout << "(AG) Performing forward propagation (slghtl mdfd) ...\n";

        cudaMalloc(&srcData, image_size*sizeof(float));
        cudaMemcpy(srcData, imgData_h,
                                    image_size*sizeof(float),
                                    cudaMemcpyHostToDevice);
//
        n = IMAGE_N; c = IMAGE_C; h = IMAGE_H; w = IMAGE_W;
        convoluteForward(conv[0], n, c, h, w, srcData, &dstData);
        normalize(dstData, n, c, h, w);
//        
        convoluteForward(conv_block[0], n, c, h, w, dstData, &srcData);
        convoluteForward(conv_block[1], n, c, h, w, srcData, &dstData);
        convoluteForward(conv_block[2], n, c, h, w, dstData, &srcData);
        normalize(srcData, n, c, h, w);

        convoluteForward(conv[1], n, c, h, w, srcData, &dstData);
        normalize(dstData, n, c, h, w);
//
        convoluteForward(conv_block[3], n, c, h, w, dstData, &srcData);
        convoluteForward(conv_block[4], n, c, h, w, srcData, &dstData);
        convoluteForward(conv_block[5], n, c, h, w, dstData, &srcData);
        normalize(srcData, n, c, h, w);
//
        convoluteForward(conv[2], n, c, h, w, srcData, &dstData);
        normalize(dstData, n, c, h, w);
//
        convoluteForward(conv_block[6], n, c, h, w, dstData, &srcData);
        convoluteForward(conv_block[7], n, c, h, w, srcData, &dstData);
        convoluteForward(conv_block[8], n, c, h, w, dstData, &srcData);
        convoluteForward(conv_block[9], n, c, h, w, srcData, &dstData);
        convoluteForward(conv_block[10], n, c, h, w, dstData, &srcData);
        convoluteForward(conv_block[11], n, c, h, w, srcData, &dstData);
        convoluteForward(conv_block[12], n, c, h, w, dstData, &srcData);
        convoluteForward(conv_block[13], n, c, h, w, srcData, &dstData);
        convoluteForward(conv_block[14], n, c, h, w, dstData, &srcData);
        normalize(srcData, n, c, h, w);
//
        convoluteForward(conv[3], n, c, h, w, srcData, &dstData);
        normalize(dstData, n, c, h, w);
//
        convoluteForward(conv_block[15], n, c, h, w, dstData, &srcData);
        convoluteForward(conv_block[16], n, c, h, w, srcData, &dstData);
        convoluteForward(conv_block[17], n, c, h, w, dstData, &srcData);
        normalize(srcData, n, c, h, w);
//
        cudaDeviceSynchronize();
//        
        printf("(AG) Result: n=%d, c=%d, h=%d, w=%d\n", n, c, h, w);
//
        value_type *dstData_h = new float[n*c*h*w];
        cudaMemcpy(dstData_h, srcData, n*c*h*w*sizeof(float), cudaMemcpyDeviceToHost);       
//
        if(TEST){
          binWrite(dstData_h, "src/output.bin", n*c*h*w);
        }
//
        cudaFree(srcData);
        cudaFree(dstData);
        delete [] dstData_h;
    }
/*
end AG mdfctn
*/


    void convNext_backbone(Layer_t<value_type>& conv1,
                              Layer_t<value_type>& conv_block1,
                              Layer_t<value_type>& conv_block2,
                              Layer_t<value_type>& conv_block3,
                              Layer_t<value_type>& conv2,
                              Layer_t<value_type>& conv_block4,
                              Layer_t<value_type>& conv_block5,
                              Layer_t<value_type>& conv_block6,
                              Layer_t<value_type>& conv3,
                              Layer_t<value_type>& conv_block7,
                              Layer_t<value_type>& conv_block8,
                              Layer_t<value_type>& conv_block9,
                              Layer_t<value_type>& conv_block10,
                              Layer_t<value_type>& conv_block11,
                              Layer_t<value_type>& conv_block12,
                              Layer_t<value_type>& conv_block13,
                              Layer_t<value_type>& conv_block14,
                              Layer_t<value_type>& conv_block15,
                              Layer_t<value_type>& conv4,
                              Layer_t<value_type>& conv_block16,
                              Layer_t<value_type>& conv_block17,
                              Layer_t<value_type>& conv_block18)
    {
        int n,c,h,w;
        int image_size = IMAGE_N*IMAGE_C*IMAGE_H*IMAGE_W;
        value_type *srcData = NULL, *dstData = NULL;
        value_type *imgData_h = new value_type[image_size*sizeof(value_type)];
        bool TEST = false;

        if(!TEST){
          binRead(imgData_h, "src/input.bin", image_size*sizeof(value_type));
        }
        else{
          generate_data(imgData_h, image_size, false);
          binWrite(imgData_h, "src/input.bin", image_size);
        }
        
        std::cout << "Performing forward propagation ...\n";

        cudaMalloc(&srcData, image_size*sizeof(value_type));
        cudaMemcpy(srcData, imgData_h,
                                    image_size*sizeof(value_type),
                                    cudaMemcpyHostToDevice);

        n = IMAGE_N; c = IMAGE_C; h = IMAGE_H; w = IMAGE_W;
        convoluteForward(conv1, n, c, h, w, srcData, &dstData);
        normalize(dstData, n, c, h, w);
        
        convoluteForward(conv_block1, n, c, h, w, dstData, &srcData);
        convoluteForward(conv_block2, n, c, h, w, srcData, &dstData);
        convoluteForward(conv_block3, n, c, h, w, dstData, &srcData);
        normalize(srcData, n, c, h, w);

        convoluteForward(conv2, n, c, h, w, srcData, &dstData);
        normalize(dstData, n, c, h, w);

        convoluteForward(conv_block4, n, c, h, w, dstData, &srcData);
        convoluteForward(conv_block5, n, c, h, w, srcData, &dstData);
        convoluteForward(conv_block6, n, c, h, w, dstData, &srcData);
        normalize(srcData, n, c, h, w);

        convoluteForward(conv3, n, c, h, w, srcData, &dstData);
        normalize(dstData, n, c, h, w);

        convoluteForward(conv_block7, n, c, h, w, dstData, &srcData);
        convoluteForward(conv_block8, n, c, h, w, srcData, &dstData);
        convoluteForward(conv_block9, n, c, h, w, dstData, &srcData);
        convoluteForward(conv_block10, n, c, h, w, srcData, &dstData);
        convoluteForward(conv_block11, n, c, h, w, dstData, &srcData);
        convoluteForward(conv_block12, n, c, h, w, srcData, &dstData);
        convoluteForward(conv_block13, n, c, h, w, dstData, &srcData);
        convoluteForward(conv_block14, n, c, h, w, srcData, &dstData);
        convoluteForward(conv_block15, n, c, h, w, dstData, &srcData);
        normalize(srcData, n, c, h, w);

        convoluteForward(conv4, n, c, h, w, srcData, &dstData);
        normalize(dstData, n, c, h, w);

        convoluteForward(conv_block16, n, c, h, w, dstData, &srcData);
        convoluteForward(conv_block17, n, c, h, w, srcData, &dstData);
        convoluteForward(conv_block18, n, c, h, w, dstData, &srcData);
        normalize(srcData, n, c, h, w);

        cudaDeviceSynchronize();
        
        printf("Result: n=%d, c=%d, h=%d, w=%d\n", n, c, h, w);

        value_type *dstData_h = new value_type[n*c*h*w];
        cudaMemcpy(dstData_h, srcData, n*c*h*w*sizeof(value_type), cudaMemcpyDeviceToHost);       

        if(TEST){
          binWrite(dstData_h, "src/output.bin", n*c*h*w);
        }
        //print(srcData,n,1,h,w);

        //printf("Result: n=%d, c=%d, h=%d, w=%d\n", n, conv_block18.outputs, h/8, w/8);
        

        cudaFree(srcData);
        cudaFree(dstData);
        delete [] dstData_h;
    }
};


int main() {
      network_t<float> ConvNext;

/*
    Blck AG mdfctn of MZ code
*/

        Layer_ag conv[4] = {
                {"conv1",CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,              3,96*2^(1-1),3, 1,1, 4,4},
                {"conv2",CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,     96*2^(1-1),96*2^(2-1),3, 1,1, 2,2},
                {"conv3",CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,     96*2^(2-1),96*2^(3-1),3, 1,1, 2,2},
                {"conv4",CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,     96*2^(3-1),96*2^(4-1),3, 1,1, 2,2}
        };
        Layer_ag conv_block[18] = {
            { "conv_block1", CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,      96*2^(1-1),96*2^(1-1),3, 1,1, 1,1},
            { "conv_block2", CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,      96*2^(1-1),96*2^(1-1),3, 1,1, 1,1},
            { "conv_block3", CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,      96*2^(1-1),96*2^(1-1),3, 1,1, 1,1},
            //
            { "conv_block4", CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,      96*2^(2-1),96*2^(2-1),3, 1,1, 1,1},
            { "conv_block5", CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, 96*2^(2-1),96*2^(2-1),3, 1,1, 1,1},                       /*WTF?*/
            { "conv_block6", CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,      96*2^(2-1),96*2^(2-1),3, 1,1, 1,1},
            //
            { "conv_block7", CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, 96*2^(3-1),96*2^(3-1),3, 1,1, 1,1},
            { "conv_block8", CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, 96*2^(3-1),96*2^(3-1),3, 1,1, 1,1},
            { "conv_block9", CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, 96*2^(3-1),96*2^(3-1),3, 1,1, 1,1},
            {"conv_block10", CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, 96*2^(3-1),96*2^(3-1),3, 1,1, 1,1},
            {"conv_block11", CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, 96*2^(3-1),96*2^(3-1),3, 1,1, 1,1},
            {"conv_block12", CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, 96*2^(3-1),96*2^(3-1),3, 1,1, 1,1},
            {"conv_block13", CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, 96*2^(3-1),96*2^(3-1),3, 1,1, 1,1},
            {"conv_block14", CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, 96*2^(3-1),96*2^(3-1),3, 1,1, 1,1},
            {"conv_block15", CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, 96*2^(3-1),96*2^(3-1),3, 1,1, 1,1},
            //
            {"conv_block16", CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,      96*2^(4-1),96*2^(4-1),3, 1,1, 1,1},
            {"conv_block17", CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,      96*2^(4-1),96*2^(4-1),3, 1,1, 1,1},
            {"conv_block18", CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,      96*2^(4-1),96*2^(4-1),3, 1,1, 1,1},
        };
/*
end AG mdfctn
*/


      std::string layer_name = "conv1";
      Layer_t<float> conv1(layer_name, (cudnnConvolutionFwdAlgo_t)0, 3,96,3,1,1,4,4);
      layer_name = "conv_block1";
      Layer_t<float> conv_block1(layer_name, (cudnnConvolutionFwdAlgo_t)6, 96,96,3,1,1,1,1);
      layer_name = "conv_block2";
      Layer_t<float> conv_block2(layer_name, (cudnnConvolutionFwdAlgo_t)6, 96,96,3,1,1,1,1);
      layer_name = "conv_block3";
      Layer_t<float> conv_block3(layer_name, (cudnnConvolutionFwdAlgo_t)6, 96,96,3,1,1,1,1);
      layer_name = "conv2";
      Layer_t<float> conv2(layer_name, (cudnnConvolutionFwdAlgo_t)1, 96,192,3,1,1,2,2);
      layer_name = "conv_block4";
      Layer_t<float> conv_block4(layer_name, (cudnnConvolutionFwdAlgo_t)6, 192,192,3,1,1,1,1);
      layer_name = "conv_block5";
      Layer_t<float> conv_block5(layer_name, (cudnnConvolutionFwdAlgo_t)1, 192,192,3,1,1,1,1);
      layer_name = "conv_block6";
      Layer_t<float> conv_block6(layer_name, (cudnnConvolutionFwdAlgo_t)6, 192,192,3,1,1,1,1);
      layer_name = "conv3";
      Layer_t<float> conv3(layer_name, (cudnnConvolutionFwdAlgo_t)1, 192,384,3,1,1,2,2);
      layer_name = "conv_block7";
      Layer_t<float> conv_block7(layer_name, (cudnnConvolutionFwdAlgo_t)1, 384,384,3,1,1,1,1);
      layer_name = "conv_block8";
      Layer_t<float> conv_block8(layer_name, (cudnnConvolutionFwdAlgo_t)1, 384,384,3,1,1,1,1);
      layer_name = "conv_block9";
      Layer_t<float> conv_block9(layer_name, (cudnnConvolutionFwdAlgo_t)1, 384,384,3,1,1,1,1);
      layer_name = "conv_block10";
      Layer_t<float> conv_block10(layer_name, (cudnnConvolutionFwdAlgo_t)1, 384,384,3,1,1,1,1);
      layer_name = "conv_block11";
      Layer_t<float> conv_block11(layer_name, (cudnnConvolutionFwdAlgo_t)1, 384,384,3,1,1,1,1);
      layer_name = "conv_block12";
      Layer_t<float> conv_block12(layer_name, (cudnnConvolutionFwdAlgo_t)1, 384,384,3,1,1,1,1);
      layer_name = "conv_block13";
      Layer_t<float> conv_block13(layer_name, (cudnnConvolutionFwdAlgo_t)1, 384,384,3,1,1,1,1);
      layer_name = "conv_block14";
      Layer_t<float> conv_block14(layer_name, (cudnnConvolutionFwdAlgo_t)1, 384,384,3,1,1,1,1);
      layer_name = "conv_block15";
      Layer_t<float> conv_block15(layer_name, (cudnnConvolutionFwdAlgo_t)1, 384,384,3,1,1,1,1);
      layer_name = "conv4";
      Layer_t<float> conv4(layer_name, (cudnnConvolutionFwdAlgo_t)2, 384,768,3,1,1,2,2);
      layer_name = "conv_block16";
      Layer_t<float> conv_block16(layer_name, (cudnnConvolutionFwdAlgo_t)6, 768,768,3,1,1,1,1);
      layer_name = "conv_block17";
      Layer_t<float> conv_block17(layer_name, (cudnnConvolutionFwdAlgo_t)6, 768,768,3,1,1,1,1);
      layer_name = "conv_block18";
      Layer_t<float> conv_block18(layer_name, (cudnnConvolutionFwdAlgo_t)6, 768,768,3,1,1,1,1);

      
      std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
      ConvNext.convNext_backbone(conv1,
                                    conv_block1,
                                    conv_block2,
                                    conv_block3,
                                    conv2,
                                    conv_block4,
                                    conv_block5,
                                    conv_block6,
                                    conv3,
                                    conv_block7,
                                    conv_block8,
                                    conv_block9,
                                    conv_block10,
                                    conv_block11,
                                    conv_block12,
                                    conv_block13,
                                    conv_block14,
                                    conv_block15,
                                    conv4,
                                    conv_block16,
                                    conv_block17,
                                    conv_block18);
      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
      std::cout << "Total warmup time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

      begin = std::chrono::steady_clock::now();
      ConvNext.convNext_backbone(conv1,
                                    conv_block1,
                                    conv_block2,
                                    conv_block3,
                                    conv2,
                                    conv_block4,
                                    conv_block5,
                                    conv_block6,
                                    conv3,
                                    conv_block7,
                                    conv_block8,
                                    conv_block9,
                                    conv_block10,
                                    conv_block11,
                                    conv_block12,
                                    conv_block13,
                                    conv_block14,
                                    conv_block15,
                                    conv4,
                                    conv_block16,
                                    conv_block17,
                                    conv_block18);
      end = std::chrono::steady_clock::now();
      std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
     
/*
    Blck AG mdfctn of MZ code
*/
      begin = std::chrono::steady_clock::now();
      ConvNext.convNext_backbone(conv, conv_block);
      end = std::chrono::steady_clock::now();
      std::cout << "(AG) Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
/*
end AG mdfctn
*/

      checkCudaErrors(cudaDeviceReset());
      return 0;
}