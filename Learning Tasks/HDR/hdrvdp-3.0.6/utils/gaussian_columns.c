#include "mex.h"
#include "math.h"
/* function which take matrix and apply discret gaussian blur on each column with given kernel and size of jump */
void mexFunction(int nlhs, mxArray *plhs[ ],int nrhs, const mxArray *prhs[ ]) 
{
    int h,i,j,w;
    
    /*input*/
    int height = mxGetM(prhs[0]);
    int width = mxGetN(prhs[0]);
    double *data = mxGetPr(prhs[0]);
    
    int kernel_len = mxGetN(prhs[1]);
    int kernel_len_2 = kernel_len/2;
    double *kernel =  mxGetPr(prhs[1]);
    
    double *jump = mxGetPr(prhs[2]);

    if(!mxIsDouble(prhs[0])){
        printf("Matrix has to contain double numbers!\n");
        plhs[0] = mxCreateDoubleMatrix(0,0,mxREAL);
        double *output = mxGetPr(plhs[0]);
        return;
    }
    
    /*output*/
    plhs[0] = mxCreateDoubleMatrix(height,width,mxREAL);
    double *output = mxGetPr(plhs[0]);
        
    /*gaussian in columns*/
    for(w=0;w < width;w++){
        for (i = 0; i < height; i++){
            int hw=height*w;
            output[hw+i]=0;
            for (j=0; j<kernel_len ;j++){
              int l=(j-kernel_len_2)*jump[0]+i;
              if(l<0) l=-l;
              if(l>=height) l=2*height-2-l;
              output[hw+i]+=kernel[j]*data[hw+l];     
     }}}
}
    
   

