#include <stdint.h>
#include <stdio.h>
#include <stdlib.h> 
#include <math.h> 
#include "timerc.h"

#define PHI 0x9e3779b9

// int ITERATIONS = INT_MAX/16;
#define ITERATIONS 1024*128
#define THRESHOLD 0.5

#define THREADSPERBLOCK 1024
#define NUMSTREAMS 16

__device__ static uint32_t Q[4096], c = 362436;

__device__ void init_rand(uint32_t x)
{
    int i;

    Q[0] = x;
    Q[1] = x + PHI;
    Q[2] = x + PHI + PHI;

    for (i = 3; i < 4096; i++)
            Q[i] = Q[i - 3] ^ Q[i - 2] ^ PHI ^ i;
}

__device__ uint32_t rand_cmwc(void)
{
    uint64_t t, a = 18782LL;
    static uint32_t i = 4095;
    uint32_t x, r = 0xfffffffe;
    i = (i + 1) & 4095;
    t = a * Q[i] + c;
    c = (t >> 32);
    x = t + c;
    if (x < c) {
            x++;
            c++;
    }
    return (Q[i] = r - x);
}

__device__ int randInRange(int min, int max, uint32_t seed) {
    return min + rand_cmwc() % (max + 1 - min);
}


/* 
* RETURNS: d, distance from  point p to the line Ax + By = C
*/
__host__ __device__ double distanceFromLine(double x, double y, double a, double b, double c) {
    double d = abs((a * x + b * y + c)) / (sqrt(a * a + b * b)); 

    return d;
}



/* 
* RETURNS: [A, B, C] for a line equation
*/
__host__ __device__ double *lineFromPoints(double *out, double x1, double y1, double x2, double y2) {
    out[0] = y1 - y2;
    out[1] = x2 - x1;
    out[2] = (x1-x2)*y1 + (y2-y1)*x1;

    return out;
}

// https://blog.csdn.net/qianhen123/article/details/80785091
// clc;
// clear all;

// %模拟输入
// src=[1,4,6;3,7,11];
// p=[src;ones(1,3)];
// m=[1.23,0.67,2.5;
//   -3.45,1.18,-4.3;
//    0,    0,    1];
// q=m*p; A=M*B

// %1.获得临时临时变量
// x=1;
// y=2;
// px01=p(x,1)-p(x,2);px12=p(x,2)-p(x,3);px02=p(x,1)-p(x,3);
// py01=p(y,1)-p(y,2);py12=p(y,2)-p(y,3);py02=p(y,1)-p(y,3);
// qx01=q(x,1)-q(x,2);qx12=q(x,2)-q(x,3);qx02=q(x,1)-q(x,3);
// qy01=q(y,1)-q(y,2);qy12=q(y,2)-q(y,3);qy02=q(y,1)-q(y,3);

// %2.计算旋转放缩因子
// det_m=px02*py12-px12*py02;
// m00=(qx01*py12-qx12*py01)/(det_m);
// m01=(qx12*px01-qx01*px12)/(det_m);
// m10=(qy01*py12-qy12*py01)/(det_m);
// m11=(qy12*px01-qy01*px12)/(det_m);

// %3.计算平移因子
// m02=q(x,1)-m00*p(x,1)-m01*p(y,1);
// m12=q(y,1)-m10*p(x,1)-m11*p(y,1);

// %4.实际输出仿射矩阵
// affine_matrix=[m00,m01,m02;
//                m10,m11,m12;
//                0,    0,  1];

__host__ __device__ double model_residual(const double * const affine_matrix, 
                                            const double& A_x, const double& A_y, 
                                            const double& B_x, const double& B_y) {

    double pre_x = B_x * affine_matrix[0] + B_y * affine_matrix[1] + affine_matrix[2];
    double pre_y = B_x * affine_matrix[3] + B_y * affine_matrix[4] + affine_matrix[5];
    double d = sqrt( (pre_x - A_x)^2 + (pre_y - A_y)^2 );//residual 
    return d;
}

// q=m*p; A=M*B


__host__ __device__ double *AffineModelFromPoints(double *affine_matrix, 
                                                  const double& A_x1, const double& src_y1, const double& src_x2, const double& src_y2, const double& src_x3, const double& src_y3,
                                                  const double& dst_x1, const double& dst_y1, const double& dst_x2, const double& dst_y2, const double& dst_x3, const double& dst_y3) {
    
    // A=MB
    double px12 = src_x1 - src_x2;
    double px13 = src_x1 - src_x3;
    double px23 = src_x2 - src_x3;
    double py12 = src_y1 - src_y2;
    double py13 = src_y1 - src_y3;
    double py23 = src_y2 - src_y3;
    double qx12 = dst_x1 - dst_x2;
    double qx13 = dst_x1 - dst_x3;
    double qx23 = dst_x2 - dst_x3;
    double qy12 = dst_y1 - dst_y2;
    double qy13 = dst_y1 - dst_y3;
    double qy23 = dst_y2 - dst_y3;
    

    double det_m=px13*py12-px12*py13;
    double m00=(qx12*py23-qx23*py12)/(det_m);
    double m01=(qx23*px12-qx12*px23)/(det_m);
    double m10=(qy12*py23-qy23*py12)/(det_m);
    double m11=(qy23*px12-qy12*px23)/(det_m);

    // %3.计算平移因子
    double m13=dst_x计算指针数组的长度1-m00*scr_x1-m12*scr_y1;
    double m12=dst_y计算指针数组的长度1-m10*scr_x1-m11*scr_y1;
计算指针数组的长度
    // %4.实际输出仿射矩阵
    affine_matrix=[m00,m12,m13;
                m10,m11,m12;
                0,    0,  1];
    

    affine_matrix[0] = y1 - y2;
    affine_matrix[1] = x2 - x1;
    affine_matrix[2] = (x1-x2)*y1 + (y2-y1)*x1;

    return out;
}


/*
* data – A set of observations.
* lineArr - Container for optimal model parameters outputted by the algorithm
* max_trials – Maximum number of iterations allowed in the algorithm.
* t – threshold value to determine data points that are fit well by model.
* d – Number of close data points required to assert that a model fits well to data.
* seed - Random seed for a RNG on device
* numStreams - Number of streams running this function. Set to 1 for testing multi-thread performance
* stream - Index of the current stream used to offset data and lineArr. Used for debugging
如果是线性模型的话　就是拟合一条直线，输入数据是[xi yi] xi是源数据　yi是目标数据 ｉ是第i条数据
如果是仿射模型的话　就是拟合一个仿射矩阵M，输入数据是[A_pt_i B_pt_i] A_pt_i是源原图像坐标B_pt_i是目标图像坐标 A=MB 求M
*/
// double *data, double *lineArr,  int k, int t, int d, uint32_t seed, int numStreams, int stream
__global__ void ransac_gpu_optimal(const double *A_Pts, const double *B_Pts, double *d_affineModel, int* maxinlines_nums_PerThread, int max_trials, 
                                    int inline_threshold, int stop_sample_num, uint32_t seed, int numStreams, int stream) {
    init_rand(seed);
    maxinlines_nums_PerThread[threadIdx.x] = 0;

    int r, inliers;
    int maxInliers = 0;
    int scopeSize = max_trials / THREADSPERBLOCK / numStreams;
    int offset = 2 * threadIdx.x * scopeSize;//scopeSize step

    double bestA, bestB, bestC, Ａ_x1, Ａ_y1, Ａ_x2, Ａ_y2, Ａ_x3, Ａ_y3, 
                                B_x1, B_y1, B_x2, B_y2, B_x3, B_y3, residual;

    // double *A_shiftedData = &A_Pts[offset];
    // double *B_shiftedData = &B_Pts[offset];


    double *line = &lineArr[threadIdx.x * 3];
    
    // 每个thread responsiable for data in scope
    for (int i=0; i < scopeSize; i++) {
        inliers = 0;

        /*******************
        CHOOSING RANDOM LINE
        *******************/

        // Choosing first random point
        r = randInRange(0, scopeSize*2 - 1, seed);
        Ａ_x1 = A_Pts[r];
        A_y1 = A_Pts[r+1];
        B_x1 = B_Pts[r];
        B_y1 = B_Pts[r+1];
        // Choosing second random point
        r = randInRange(0, scopeSize*2 - 1, seed);
        Ａ_x2 = A_Pts[r];
        Ａ_y2 = A_Pts[r+1];
        B_x2 = B_Pts[r];
        B_y2 = B_Pts[r+1];
        // Choosing second random point
        r = randInRange(0, scopeSize*2 - 1, seed);
        Ａ_x3 = A_Pts[r];
        Ａ_y3 = A_Pts[r+1];
        B_x3 = B_Pts[r];
        B_y3 = B_Pts[r+1];

        // Modeling a line between those two points
        // line = lineFromPoints(line, x1, y1, x2, y2);
        AffineModelFromPoints(d_affineModel, Ａ_x1, Ａ_y1, Ａ_x2, Ａ_y2, Ａ_x3, Ａ_y3, 
                                B_x1, B_y1, B_x2, B_y2, B_x3, B_y3);

        /***********************
        FINDING INLIERS FOR LINE
        ***********************/
        for (int j=0; j < scopeSize*2; j=j+2) {
            Ａ_x1 = A_Pts[j];
            Ａ_y1 = A_Pts[j + 1];
            B_x1 = B_Pts[j];
            B_y1 = B_Pts[j + 1];
            // dist = distanceFromLine(x1, y1, line[0], line[1], line[2]);
            residual = model_residual(d_affineModel, Ａ_x1, Ａ_y1, B_x1, B_y1);
            if (residual <= inline_threshold) {
                inliers++;
            }
        }
        
        if (inliers > maxInliers) {
            maxInliers = inliers;
            bestA = line[0];
            bestB = line[1];
            bestC = line[2];
        }

        // if (maxInliers >= ( stop_sample_num / THREADSPERBLOCK)) {
        //     break;
        // }
    }

    // Some reduction
    // if (bestA == -bestB) {
    //     bestA = 1;
    //     bestB = -1;
    // }

    // lineArr[threadIdx.x * 3] = bestA;
    // lineArr[threadIdx.x * 3 + 1] = bestB;
    // lineArr[threadIdx.x * 3 + 2] = bestC;

    // Print out only some of the output to check correctness
    // if (threadIdx.x == 0 && stream % 4 == 0) {
    //     printf("GPU w/ Streams: A=%f | B=%f | C=%f \n", bestA, bestB, bestC);
    // }
}





void ransac_cpu(double *data, double *line, int k, int t, int d){
    srand(time(NULL)); 

    int r, inliers;
    int maxInliers = 0;

    double bestA, bestB, bestC, x1, y1, x2, y2, dist;
    
    for (int i=0; i < k; i++) {
        inliers = 0;

        /*******************
        CHOOSING RANDOM LINE
        *******************/

        // Choosing first random point
        r = 1 + rand() % k;

        x1 = data[r];
        y1 = data[r+1];

        // Choosing second random point
        r = 1 + rand() % k;

        x2 = data[r];
        y2 = data[r+1];

        // Modeling a line between those two points
        line = lineFromPoints(line, x1, y1, x2, y2);


        /***********************
        FINDING INLIERS FOR LINE
        ***********************/
        for (int j=0; j < k; j=j+2) {
            x1 = data[j*2];
            y1 = data[j*2 + 1];
            dist = distanceFromLine(x1, y1, line[0], line[1], line[2]);
            if (dist <= t) {
                inliers++;
            }
        }
        
        if (inliers > maxInliers) {
            maxInliers = inliers;
            bestA = line[0];
            bestB = line[1];
            bestC = line[2];
        }

        if (maxInliers >= d) 
            break;
    }

    // Some reduction
    if (bestA == -bestB) {
        bestA = 1;
        bestB = -1;
    }

    line[0] = bestA;
    line[1] = bestB;
    line[2] = bestC;
}



int ransac_gpu(double *Ａ_points, double *B_points, 
             const char* model, int min_samples=3, float residual_threshold=10, max_trials=1024){

// max_trials需要是1024的倍数

if (strlen(Ａ_points) != strlen(B_points)){
    return;
}

    int matched_pts = strlen(Ａ_points);
    double *d_A_points, *d_B_points;
    cudaMalloc((void **) &d_A_points, (2*matched_pts*sizeof(double)));
    cudaMalloc((void **) &d_B_points, (2*matched_pts*sizeof(double)));

    cudaMemcpy(d_A_points, Ａ_points, (2*matched_pts*sizeof(double)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_points, B_points, (2*matched_pts*sizeof(double)), cudaMemcpyHostToDevice);

    // model parameter
    double *affineModel;
    double *d_affineModel;
    // Each thread will need it's own line equation container
    affineModel = (double *) malloc(9*THREADSPERBLOCK*sizeof(double));
    cudaMalloc((void **) &d_affineModel, (9*THREADSPERBLOCK*sizeof(double)));

    // 每个thread都有一个最好的inlines
    int *inlines_nums;
    int *d_inlines_nums;
    // Each thread will need it's own line equation container
    inlines_nums = (int *) malloc(THREADSPERBLOCK*sizeof(int));
    cudaMalloc((void **) &d_inlines_nums, (THREADSPERBLOCK*sizeof(int)));
    cudaMemcpy(d_inlines_nums, inlines_nums, (THREADSPERBLOCK*sizeof(int)), cudaMemcpyHostToDevice);

    ransac_gpu_optimal<<<1,THREADSPERBLOCK>>>(d_A_points, d_B_points, d_affineModel, maxinlines_nums_PerThread, max_trials, residual_threshold, pass / THREADSPERBLOCK, seed, 1, 0);

    cudaMemcpy(inlines_nums, d_inlines_nums, (THREADSPERBLOCK*sizeof(int)), cudaMemcpyDeviceToHost);
    cudaMemcpy(affineModel, d_affineModel, (9*THREADSPERBLOCK*sizeof(double)), cudaMemcpyDeviceToHost);
    
    for (int j=0; j < k; j=j+2) {
            x1 = data[j*2];
            y1 = data[j*2 + 1];
            dist = distanceFromLine(x1, y1, line[0], line[1], line[2]);
            if (dist <= t) {
                inliers++;
            }
    }

}


int main() {
    uint32_t seed = time(NULL);
    int match_pts_num = 100;//1024

    srand(seed); 
    int r;

    int pass = ITERATIONS / 2;//大于所有数据的1/2的话

    /*
    * Every two elements corresponds to x,y at time t.
    * 准备数据
    */
    // double *Ａ_points = (double *) malloc(2*match_pts_num*sizeof(double));
    // double *d_A_points;
    // double *Ａ_points = (double *) malloc(2*match_pts_num*sizeof(double));
    // double *d_A_points;

    // // Move points with velocity (vx, vy)
    // double vx = 100.0;
    // double vy = 100.0;


    // for (int j=0; j < ITERATIONS; j++) {
    //     if (j % 10 == 0) {
    //         r = 0 + rand() % ITERATIONS;
    //         points[j*2] = r;
    //         r = 0 + rand() % ITERATIONS;
    //         points[j*2+1] = r;
    //     } else {
    //         points[j*2] = j-1 + vx;
    //         points[j*2+1] = j-1 + vy; 
    //     }
    // }
    
    // Shell to be used for outputting results in the form of line equation
    double *line = (double *) malloc(3*sizeof(double));

    // Copy points to file
    FILE *fp;
    fp = fopen("p.txt", "w+");
    
    for (int i=0; i < 2*ITERATIONS; i++) {
        fprintf(fp,"%f ", points[i]);
    }
    fclose(fp);



    float cpu_time;
    cstart();
    ransac_cpu(points, line, ITERATIONS, THRESHOLD, pass);
    cend(&cpu_time);

    printf("CPU: A=%f | B=%f | C=%f \n", line[0], line[1], line[2]);
    puts("***");

    

    
    // cudaMalloc((void **) &d_points, (2*ITERATIONS*sizeof(double)));
    // cudaMemcpy(d_points, points, (2*ITERATIONS*sizeof(double)), cudaMemcpyHostToDevice);

    double *affineModel;
    double *d_affineModel;

    // Each thread will need it's own line equation container
    affineModel = (double *) malloc(9*THREADSPERBLOCK*sizeof(double));

    cudaMalloc((void **) &d_affineModel, (9*THREADSPERBLOCK*sizeof(double)));
    cudaMemcpy(d_affineModel, affineModel, (9*THREADSPERBLOCK*sizeof(double)), cudaMemcpyHostToDevice);

    float gpu_multi_thread_time;
    gstart();
    ransac_gpu_optimal<<<1,THREADSPERBLOCK>>>(d_points, d_affineModel, ITERATIONS, THRESHOLD, pass / THREADSPERBLOCK, seed, 1, 0);
    gend(&gpu_multi_thread_time);

    cudaMemcpy(affineModel, d_affineModel, (9*THREADSPERBLOCK*sizeof(double)), cudaMemcpyDeviceToHost);

    // 最后将结果汇总　平均
    double avgA = 0;
    double avgB = 0;
    double avgC = 0;
    for (int i=0; i<3*THREADSPERBLOCK; i=i+3) {
        avgA = avgA + lineArr[i];
        avgB = avgB + lineArr[i+1];
        avgC = avgC + lineArr[i+2];
    }
    avgA = avgA / THREADSPERBLOCK;
    avgB = avgB / THREADSPERBLOCK;
    avgC = avgC / THREADSPERBLOCK;

    printf("GPU w/Threads: A=%f | B=%f | C=%f \n", avgA, avgB, avgC);

    puts("***");
    cudaDeviceSynchronize();





    double *lineStreamArr;
    double *d_lineStreamArr;

    // Each thread will need it's only line equation container
    lineStreamArr = (double *) malloc(3*NUMSTREAMS*THREADSPERBLOCK*sizeof(double));
    cudaMalloc((void **) &d_lineStreamArr, (3*NUMSTREAMS*THREADSPERBLOCK*sizeof(double)));

    int streamSize = (2 * ITERATIONS) / NUMSTREAMS;
    int streamBytes = streamSize * sizeof(double);

    cudaStream_t stream[NUMSTREAMS];

    for (int i = 0; i < NUMSTREAMS; ++i)
        cudaStreamCreate(&stream[i]);

    float gpu_stream_time;
    gstart();
    for (int i=0; i < NUMSTREAMS; i++) {
        int offset = i * streamSize;
        int lineOffset = 3 * i * THREADSPERBLOCK;
        cudaMemcpyAsync(&d_points[offset], &points[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(&d_lineStreamArr[lineOffset], &lineStreamArr[lineOffset], 3*THREADSPERBLOCK*sizeof(double), cudaMemcpyHostToDevice, stream[i]);
        ransac_gpu_optimal<<<1, THREADSPERBLOCK, 0, stream[i]>>>(&d_points[offset], &d_lineStreamArr[lineOffset], ITERATIONS, THRESHOLD, pass / THREADSPERBLOCK, seed, NUMSTREAMS, i);
    }
    gend(&gpu_stream_time);

    cudaDeviceSynchronize();

    for (int i = 0; i < NUMSTREAMS; ++i)
        cudaStreamDestroy(stream[i]);

    cudaMemcpy(lineStreamArr, d_lineStreamArr, (3*NUMSTREAMS*THREADSPERBLOCK*sizeof(double)), cudaMemcpyDeviceToHost);

    avgA = 0;
    avgB = 0;
    avgC = 0;
    for (int i=0; i<3*NUMSTREAMS*THREADSPERBLOCK; i=i+3) {
        avgA = avgA + lineStreamArr[i];
        avgB = avgB + lineStreamArr[i+1];
        avgC = avgC + lineStreamArr[i+2];
    }
    avgA = avgA / THREADSPERBLOCK / NUMSTREAMS;
    avgB = avgB / THREADSPERBLOCK / NUMSTREAMS;
    avgC = avgC / THREADSPERBLOCK / NUMSTREAMS;

    printf("GPU w/Streams: A=%f | B=%f | C=%f \n", avgA, avgB, avgC);

    // for (int b=0; b<9; b=b+3) {
    //     printf("GPU w/Streams: A=%f | B=%f | C=%f \n", lineStreamArr[b], lineStreamArr[b+1], lineStreamArr[b+2]);
    // }
    puts("***\n");
    cudaDeviceSynchronize();




    printf("CPU time: %f\n",cpu_time);
    printf("GPU w/ Multi-thread time: %f\n", gpu_multi_thread_time);
    printf("GPU w/ Streams time: %f\n", gpu_stream_time);



    cudaFree(d_points);
    cudaFree(d_lineArr);
    cudaFree(d_lineStreamArr);
    free(points);
    free(line);
    free(lineArr);
    free(lineStreamArr);

    return 0;
}