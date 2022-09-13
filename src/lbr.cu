#include <cuda.h>
#include "cublas_v2.h"
#include <cudnn.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

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

/*

*/
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
                        bool TEST = true;
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
/*

*/
//#include "jlcxx/jlcxx.hpp"
//
//std::string greet()
//{
//   return "hello, world";
//}
//extern int lib_fnctn( void )
//{
//  std::cout << "\nThis function seems to work" << 1 << "\n"
//    "OpenCV version " << CV_VERSION << std::endl;
//
//  return 1;
//}
//JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
//{
//  mod.method("greet", &greet);
//  mod.method("lib_fnctn", &lib_fnctn);
//}
//

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
    void convoluteBlock(Layer_ag &conv,
                        int& n, int& c, int& h, int& w,
                        float* srcData, float** dstData)
    {
        Layer_ag tmp = conv;
        conv.str_h = 1;
        conv.str_w = 1;
        conv.pad_h = 3;
        conv.pad_w = 3;
        convoluteForward(conv, n,c,h,w, srcData, dstData);
        normalize(*dstData, n, c, h, w);
        conv.str_h = 4;
        conv.str_w = 4;
        conv.pad_h = 0;
        conv.pad_w = 0;
        conv.kernel_dim = 1;
        conv.outputs = conv.outputs*4;
        convoluteForward(conv, n,c,h,w, *dstData, &srcData);
        print(srcData,1,1,5,5);
        print(*dstData,1,1,5,5);
        conv.str_h = 1;
        conv.str_w = 1;
        conv.pad_h = 0;
        conv.pad_w = 0;
        conv.kernel_dim = 1;
        conv.inputs = conv.outputs;
        conv.outputs = (int)conv.outputs/4;
        printf("(AG) should be: n=%d, out=%d, in=%d, w=%d\n", n, conv.outputs, conv.inputs, w);
        printf("(AG) tmp    be: n=%d, out=%d, in=%d, w=%d\n", n, tmp.outputs, tmp.inputs, w);
        convoluteForward(conv, n,c,h,w, srcData, dstData);
        conv = tmp;
    }
    void convoluteForward(const Layer_ag &conv,
                          int& n, int& c, int& h, int& w,
                          float* srcData, float** dstData)
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
//        std::cout << "test conv block 1\n";
        convoluteForward(conv_block[0], n, c, h, w, dstData, &srcData);
//        convoluteBlock(conv_block[0], n, c, h, w, dstData, &srcData);
        convoluteForward(conv_block[1], n, c, h, w, dstData, &srcData); //srcData, &dstData);
//        std::cout << "test conv_block 2\n";
//        convoluteBlock(conv_block[1], n, c, h, w, srcData, &dstData); //srcData, &dstData);
//        std::cout << "test conv_block 3\n";
        convoluteForward(conv_block[2], n, c, h, w, srcData, &dstData); //dstData, &srcData);
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
//        printf("(AG) Result: n=%d, c=%d, h=%d, w=%d\n", n, c, h, w);
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


};

