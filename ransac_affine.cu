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
    // q=m*p; A=M*B
    double pre_x = B_x * affine_matrix[0] + B_y * affine_matrix[1] + affine_matrix[2];
    double pre_y = B_x * affine_matrix[3] + B_y * affine_matrix[4] + affine_matrix[5];
    double d = sqrt( (pre_x - A_x)*(pre_x - A_x) + (pre_y - A_y)*(pre_y - A_y) );//residual
    return d;
}



// 计算仿射变换矩阵系数
__host__ __device__ double *AffineModelFromPoints(double *affine_matrix,
                                                  const double& A_x1, const double& A_y1, const double& A_x2, const double& A_y2, const double& A_x3, const double& A_y3,
                                                  const double& B_x1, const double& B_y1, const double& B_x2, const double& B_y2, const double& B_x3, const double& B_y3) {

    // q=m*p; A=M*B
    double px12 = B_x1 - B_x2;
    double px13 = B_x1 - B_x3;
    double px23 = B_x2 - B_x3;
    double py12 = B_y1 - B_y2;
    double py13 = B_y1 - B_y3;
    double py23 = B_y2 - B_y3;
    double qx12 = A_x1 - A_x2;
    double qx13 = A_x1 - A_x3;
    double qx23 = A_x2 - A_x3;
    double qy12 = A_y1 - A_y2;
    double qy13 = A_y1 - A_y3;
    double qy23 = A_y2 - A_y3;


    // %2.计算旋转放缩因子
    double det_p=px13*py23-px23*py13;
    double m00=(qx12*py23-qx23*py12)/(det_p);
    double m01=(qx23*px12-qx12*px23)/(det_p);
    double m10=(qy12*py23-qy23*py12)/(det_p);
    double m11=(qy23*px12-qy12*px23)/(det_p);

    // %3.计算平移因子
    double m02 = A_x1 - m00 * B_x1 - m01 * B_y1;
    double m12 = A_y1 - m10 * B_x1 - m11 * B_y1;


    // %4.实际输出仿射矩阵
    // affine_matrix=[m00,m12,m13;
    //             m10,m11,m12;
    //             0,    0,  1];

    affine_matrix[0] = m00;
    affine_matrix[1] = m01;
    affine_matrix[2] = m02;
    affine_matrix[3] = m10;
    affine_matrix[4] = m11;
    affine_matrix[5] = m12;
    affine_matrix[6] = 0;
    affine_matrix[7] = 0;
    affine_matrix[8] = 1;

    return affine_matrix;
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
// d_A_points, d_B_points, matched_pts, scopeSize, inline_threshold, stop_sample_num, seed, d_affineModel, maxinlines_nums_PerThread
__global__ void ransac_gpu_optimal(const double *A_Pts, const double *B_Pts,
                                   int matched_pts, int scopeSize, int inline_threshold, int stop_sample_num, uint32_t seed,
                                   double *d_affineModel_Arr, int* maxinlines_nums_PerThread) {
    init_rand(seed);
    maxinlines_nums_PerThread[threadIdx.x] = 0;

    int r, inliers;
    int maxInliers = 0;
    // int scopeSize = max_trials / THREADSPERBLOCK / numStreams;
    // int offset = 2 * threadIdx.x * scopeSize;//scopeSize step

    double bestA, bestB, bestC, A_x1, A_y1, A_x2, A_y2, A_x3, A_y3,B_x1, B_y1, B_x2, B_y2, B_x3, B_y3, residual;

    // double *A_shiftedData = &A_Pts[offset];
    // double *B_shiftedData = &B_Pts[offset];


    double *d_affineModel = &d_affineModel_Arr[threadIdx.x * 9];

    // 每个thread responsiable for data in scope
    for (int i=0; i < scopeSize; i++) {
        inliers = 0;

        /*******************
        CHOOSING RANDOM LINE
        *******************/

        // Choosing first random point
        r = randInRange(0, 2*matched_pts - 1, seed);
        // printf("r: %d \n", r);
        A_x1 = A_Pts[r];
        A_y1 = A_Pts[r+1];
        B_x1 = B_Pts[r];
        B_y1 = B_Pts[r+1];
        // Choosing second random point
        r = randInRange(0, 2*matched_pts - 1, seed);
        // printf("r: %d \n", r);
        A_x2 = A_Pts[r];
        A_y2 = A_Pts[r+1];
        B_x2 = B_Pts[r];
        B_y2 = B_Pts[r+1];
        // Choosing second random point
        r = randInRange(0, 2*matched_pts - 1, seed);
        // printf("r: %d \n", r);
        A_x3 = A_Pts[r];
        A_y3 = A_Pts[r+1];
        B_x3 = B_Pts[r];
        B_y3 = B_Pts[r+1];

        // Modeling a line between those two points
        // line = lineFromPoints(line, x1, y1, x2, y2);
        // printf("start get model \n");
        d_affineModel = AffineModelFromPoints(d_affineModel, A_x1, A_y1, A_x2, A_y2, A_x3, A_y3,
                                B_x1, B_y1, B_x2, B_y2, B_x3, B_y3);
        
        /***********************
        FINDING INLIERS FOR LINE
        ***********************/
    //    printf("start calculate residual \n");
        for (int j=0; j < 2*matched_pts; j=j+2) {
            A_x1 = A_Pts[j];
            A_y1 = A_Pts[j + 1];
            B_x1 = B_Pts[j];
            B_y1 = B_Pts[j + 1];
            // dist = distanceFromLine(x1, y1, line[0], line[1], line[2]);
            residual = model_residual(d_affineModel, A_x1, A_y1, B_x1, B_y1);
            // if(threadIdx.x==56)
            //     printf("%d residual:%f A_x1:%f B_y1:%f\n",threadIdx.x, residual, A_x1, B_y1);
            if (residual <= 20) {
                inliers++;
            }
        }

        if (inliers > maxInliers) {
            // printf("inlines %d \n", inliers);
            maxInliers = inliers;
            // bestA = line[0];
            // bestB = line[1];
            // bestC = line[2];
        }

        

        // if (maxInliers >= ( stop_sample_num / THREADSPERBLOCK)) {
        //     break;
        // }
    }
    
    maxinlines_nums_PerThread[threadIdx.x] = maxInliers;
    printf("maxinlines_nums_PerThread[threadIdx.x]: %d = maxInliers: %d \n", maxinlines_nums_PerThread[threadIdx.x] , maxInliers);

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

    // __syncthreads();

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



int ransac_gpu(double *A_points, double *B_points, const int matched_pts,
             int min_samples=3, float inline_threshold=20, int max_trials=4096){

    if (max_trials % 32 != 0){
        return -1;
    }

    // 数据数量很少
    int threads_nums = max_trials <= THREADSPERBLOCK ? max_trials:THREADSPERBLOCK;
    
    printf("threads_nums: %d \n", threads_nums);

    int scopeSize = max_trials / threads_nums ;/// numStreams
    printf("scopeSize: %d \n", scopeSize);

    double *d_A_points, *d_B_points;
    cudaMalloc((void **) &d_A_points, (2*matched_pts*sizeof(double)));
    cudaMalloc((void **) &d_B_points, (2*matched_pts*sizeof(double)));

    cudaMemcpy(d_A_points, A_points, (2*matched_pts*sizeof(double)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_points, B_points, (2*matched_pts*sizeof(double)), cudaMemcpyHostToDevice);

    // model parameter
    double *affineModel;
    double *d_affineModel;
    // Each thread will need it's own line equation container
    affineModel = (double *) malloc(9 * threads_nums * sizeof(double));
    cudaMalloc((void **) &d_affineModel, (9 * threads_nums * sizeof(double)));

    // 每个thread都有一个最好的inlines
    int *maxinlines_nums_PerThread;
    int *d_maxinlines_nums_PerThread;
    // Each thread will need it's own line equation container
    maxinlines_nums_PerThread = (int *) malloc(threads_nums * sizeof(int));
    cudaMalloc((void **) &d_maxinlines_nums_PerThread, (threads_nums * sizeof(int)));
    cudaMemcpy(d_maxinlines_nums_PerThread, maxinlines_nums_PerThread, (threads_nums * sizeof(int)), cudaMemcpyHostToDevice);


    int stop_sample_num = 8*matched_pts/10;
    uint32_t seed = time(NULL);
    printf("stop_sample_num: %d \n", stop_sample_num);
    ransac_gpu_optimal<<<1, threads_nums>>>(d_A_points, d_B_points, matched_pts, scopeSize, inline_threshold, stop_sample_num, seed, d_affineModel, d_maxinlines_nums_PerThread);

    cudaMemcpy(maxinlines_nums_PerThread, d_maxinlines_nums_PerThread, (threads_nums * sizeof(int)), cudaMemcpyDeviceToHost);
    cudaMemcpy(affineModel, d_affineModel, (9 * threads_nums * sizeof(double)), cudaMemcpyDeviceToHost);

    // 依据内点 找出最好的模型
    int max_inlines_nums = 0;
    for (int j=0; j < threads_nums; ++j) {
        // x1 = data[j*2];
        // y1 = data[j*2 + 1];
        // dist = distanceFromLine(x1, y1, line[0], line[1], line[2]);
        // if (dist <= t) {
        //     inliers++;
        // }
        if (max_inlines_nums < maxinlines_nums_PerThread[j])
                max_inlines_nums = maxinlines_nums_PerThread[j];
        printf("maxinlines_nums_PerThread[j]: %d \n", maxinlines_nums_PerThread[j]);
    }
    printf("max_inlines_nums: %d \n", max_inlines_nums);


    cudaFree(d_A_points);
    cudaFree(d_B_points);
    cudaFree(d_affineModel);
    cudaFree(d_maxinlines_nums_PerThread);
    free(affineModel);
    free(maxinlines_nums_PerThread);

    return max_inlines_nums;
}




#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAX_LINE 1024
int read_data(const char* file_path, double *A_points, double *B_points)
{
    char buf[MAX_LINE];  /*缓冲区*/
    FILE *fp;            /*文件指针*/
    int len;             /*行字符个数*/
    const char *separator = " ";
    if((fp = fopen(file_path,"r")) == NULL)
    {
        perror("fail to read");
        exit (1) ;
    }

    int line = 0;
    while(fgets(buf, MAX_LINE,fp) != NULL)
    {
        len = strlen(buf);
        buf[len-1] = '\0';  /*去掉换行符*/
        printf("%s %d \n",buf,len - 1);
        char *pNext;
        int count = 0;
        if (buf == NULL || strlen(buf) == 0) //如果传入的地址为空或长度为0，直接终止 
            return 0;
        if (separator == NULL || strlen(separator) == 0) //如未指定分割的字符串，直接终止 
            return 0;
        pNext = (char *)strtok(buf,separator); //必须使用(char *)进行强制类型转换(虽然不写有的编译器中不会出现指针错误)
        // printf(" %s %d", pNext, atoi(pNext));
        //  while(pNext != NULL) {
        //       *dest++ = pNext;
        //       ++count;
        //      pNext = (char *)strtok(NULL,separator);  //必须使用(char *)进行强制类型转换
        // }  
        A_points[2*line] = atof(pNext);
        // printf(" %f %f \n", atof(pNext), A_points[2*line]);
        
        pNext = (char *)strtok(NULL,separator);  //必须使用(char *)进行强制类型转换
        // printf(" %s %d", pNext, atoi(pNext));
        A_points[2*line+1] = atof(pNext);
        
        pNext = (char *)strtok(NULL,separator);  //必须使用(char *)进行强制类型转换
        // printf(" %s %d", pNext, atoi(pNext));
        B_points[2*line] = atof(pNext);

        pNext = (char *)strtok(NULL,separator);  //必须使用(char *)进行强制类型转换
        // printf(" %s %d\n", pNext, atoi(pNext));
        B_points[2*line+1] = atof(pNext);

        printf("%f %f %f %f  \n",A_points[2*line],A_points[2*line+1],B_points[2*line],B_points[2*line+1]);


        line += 1;
    }
    return line;
}



int main(int argc, char **argv) {
    if(argc !=2){
        printf("please input filename\n");
        return 0;
    }

    char * filename = argv[1];

    const int matched_pts=182;
    double *A_points = (double *)malloc(2*matched_pts*sizeof(double));
    double *B_points = (double *)malloc(2*matched_pts*sizeof(double));

    char *path = "/media/liesmars/67038e2e-f9b3-41a0-b779-e53a1ca1fd8a1/scene_pic/pic-web-service/src/streetView_index/utils/test_data/6.txt";
    int match_pts = read_data(filename, A_points, B_points);
     printf("match pts: %d \n", match_pts);
    float gpu_multi_thread_time;
    gstart();


    ransac_gpu(A_points, B_points, match_pts);
        gend(&gpu_multi_thread_time);
    printf("GPU w/ Multi-thread time: %f\n", gpu_multi_thread_time);
    
    free(A_points);
    free(B_points);
    


    




//     uint32_t seed = time(NULL);
//     int match_pts_num = 100;//1024

//     srand(seed);
//     int r;

//     int pass = ITERATIONS / 2;//大于所有数据的1/2的话

//     /*
//     * Every two elements corresponds to x,y at time t.
//     * 准备数据
//     */
//     // double *A_points = (double *) malloc(2*match_pts_num*sizeof(double));
//     // double *d_A_points;
//     // double *A_points = (double *) malloc(2*match_pts_num*sizeof(double));
//     // double *d_A_points;

//     // // Move points with velocity (vx, vy)
//     // double vx = 100.0;
//     // double vy = 100.0;


//     // for (int j=0; j < ITERATIONS; j++) {
//     //     if (j % 10 == 0) {
//     //         r = 0 + rand() % ITERATIONS;
//     //         points[j*2] = r;
//     //         r = 0 + rand() % ITERATIONS;
//     //         points[j*2+1] = r;
//     //     } else {
//     //         points[j*2] = j-1 + vx;
//     //         points[j*2+1] = j-1 + vy;
//     //     }
//     // }

//     // Shell to be used for outputting results in the form of line equation
//     double *line = (double *) malloc(3*sizeof(double));

//     // Copy points to file
//     FILE *fp;
//     fp = fopen("p.txt", "w+");

//     for (int i=0; i < 2*ITERATIONS; i++) {
//         fprintf(fp,"%f ", points[i]);
//     }
//     fclose(fp);



//     float cpu_time;
//     cstart();
//     ransac_cpu(points, line, ITERATIONS, THRESHOLD, pass);
//     cend(&cpu_time);

//     printf("CPU: A=%f | B=%f | C=%f \n", line[0], line[1], line[2]);
//     puts("***");




//     // cudaMalloc((void **) &d_points, (2*ITERATIONS*sizeof(double)));
//     // cudaMemcpy(d_points, points, (2*ITERATIONS*sizeof(double)), cudaMemcpyHostToDevice);

//     double *affineModel;
//     double *d_affineModel;

//     // Each thread will need it's own line equation container
//     affineModel = (double *) malloc(9*THREADSPERBLOCK*sizeof(double));

//     cudaMalloc((void **) &d_affineModel, (9*THREADSPERBLOCK*sizeof(double)));
//     cudaMemcpy(d_affineModel, affineModel, (9*THREADSPERBLOCK*sizeof(double)), cudaMemcpyHostToDevice);

//     float gpu_multi_thread_time;
//     gstart();
//     ransac_gpu_optimal<<<1,THREADSPERBLOCK>>>(d_points, d_affineModel, ITERATIONS, THRESHOLD, pass / THREADSPERBLOCK, seed, 1, 0);
//     gend(&gpu_multi_thread_time);

//     cudaMemcpy(affineModel, d_affineModel, (9*THREADSPERBLOCK*sizeof(double)), cudaMemcpyDeviceToHost);

//     // 最后将结果汇总　平均
//     double avgA = 0;
//     double avgB = 0;
//     double avgC = 0;
//     for (int i=0; i<3*THREADSPERBLOCK; i=i+3) {
//         avgA = avgA + lineArr[i];
//         avgB = avgB + lineArr[i+1];
//         avgC = avgC + lineArr[i+2];
//     }
//     avgA = avgA / THREADSPERBLOCK;
//     avgB = avgB / THREADSPERBLOCK;
//     avgC = avgC / THREADSPERBLOCK;

//     printf("GPU w/Threads: A=%f | B=%f | C=%f \n", avgA, avgB, avgC);

//     puts("***");
//     cudaDeviceSynchronize();





//     double *lineStreamArr;
//     double *d_lineStreamArr;

//     // Each thread will need it's only line equation container
//     lineStreamArr = (double *) malloc(3*NUMSTREAMS*THREADSPERBLOCK*sizeof(double));
//     cudaMalloc((void **) &d_lineStreamArr, (3*NUMSTREAMS*THREADSPERBLOCK*sizeof(double)));

//     int streamSize = (2 * ITERATIONS) / NUMSTREAMS;
//     int streamBytes = streamSize * sizeof(double);

//     cudaStream_t stream[NUMSTREAMS];

//     for (int i = 0; i < NUMSTREAMS; ++i)
//         cudaStreamCreate(&stream[i]);

//     float gpu_stream_time;
//     gstart();
//     for (int i=0; i < NUMSTREAMS; i++) {
//         int offset = i * streamSize;
//         int lineOffset = 3 * i * THREADSPERBLOCK;
//         cudaMemcpyAsync(&d_points[offset], &points[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
//         cudaMemcpyAsync(&d_lineStreamArr[lineOffset], &lineStreamArr[lineOffset], 3*THREADSPERBLOCK*sizeof(double), cudaMemcpyHostToDevice, stream[i]);
//         ransac_gpu_optimal<<<1, THREADSPERBLOCK, 0, stream[i]>>>(&d_points[offset], &d_lineStreamArr[lineOffset], ITERATIONS, THRESHOLD, pass / THREADSPERBLOCK, seed, NUMSTREAMS, i);
//     }
//     gend(&gpu_stream_time);

//     cudaDeviceSynchronize();

//     for (int i = 0; i < NUMSTREAMS; ++i)
//         cudaStreamDestroy(stream[i]);

//     cudaMemcpy(lineStreamArr, d_lineStreamArr, (3*NUMSTREAMS*THREADSPERBLOCK*sizeof(double)), cudaMemcpyDeviceToHost);

//     avgA = 0;
//     avgB = 0;
//     avgC = 0;
//     for (int i=0; i<3*NUMSTREAMS*THREADSPERBLOCK; i=i+3) {
//         avgA = avgA + lineStreamArr[i];
//         avgB = avgB + lineStreamArr[i+1];
//         avgC = avgC + lineStreamArr[i+2];
//     }
//     avgA = avgA / THREADSPERBLOCK / NUMSTREAMS;
//     avgB = avgB / THREADSPERBLOCK / NUMSTREAMS;
//     avgC = avgC / THREADSPERBLOCK / NUMSTREAMS;

//     printf("GPU w/Streams: A=%f | B=%f | C=%f \n", avgA, avgB, avgC);

//     // for (int b=0; b<9; b=b+3) {
//     //     printf("GPU w/Streams: A=%f | B=%f | C=%f \n", lineStreamArr[b], lineStreamArr[b+1], lineStreamArr[b+2]);
//     // }
//     puts("***\n");
//     cudaDeviceSynchronize();




//     printf("CPU time: %f\n",cpu_time);
//     printf("GPU w/ Multi-thread time: %f\n", gpu_multi_thread_time);
//     printf("GPU w/ Streams time: %f\n", gpu_stream_time);



//     cudaFree(d_points);
//     cudaFree(d_lineArr);
//     cudaFree(d_lineStreamArr);
//     free(points);
//     free(line);
//     // free(lineArr);
//     free(lineStreamArr);

//     return 0;
}
