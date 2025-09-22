/**************************************************************************//**
 * @file     main.c
 * @version  V3.00
 * @brief    Perform A/D Conversion with ADC single cycle scan mode (3 channels).
 *
 * SPDX-License-Identifier: Apache-2.0
 * @copyright (C) 2018 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include <stdio.h>
#include "NuMicro.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PLL_CLOCK       50000000

/******************************************************************
 * dataset format setting
 ******************************************************************/

#define train_data_num  			180	//Total number of training data
#define test_data_num 				0	//Total number of testing data

/******************************************************************
 * Network Configuration - customized per network
 ******************************************************************/
#define input_length                    6// The number of input 
#define HiddenNodes_1                    11 // The number of neurons in hidden layer
#define HiddenNodes_2                    11 // The number of neurons in hidden layer

#define target_num                     4 // The number of output 

volatile uint32_t g_u32AdcIntFlag;
const float LearningRate =       1e-3   ;    // Learning Rate
const float Momentum = 0.9;
const float InitialWeightMax = 0.5;     // Maximum initial weight
const float goal_acc =  0.95          ;    // Target accuracy

// Create train dataset/output
float train_data_input[train_data_num][input_length] = {
    4.07 , -0.56 , 0.56 , 3.88 , -0.37 , 0.55 , 
 4.47 , -1.59 , 1.13 , 3.11 , 1.30 , -1.67 , 
 3.78 , 0.98 , 0.05 , 4.02 , -2.13 , 0.80 , 
 3.94 , -1.50 , 0.77 , 3.91 , -0.80 , 0.29 , 
 3.93 , 0.30 , 0.30 , 3.42 , -1.05 , 0.70 ,
 3.64 , 0.02 , 0.48 , 3.05 , -0.35 , 0.58 , 
 5.49 , 0.79 , 2.80 , 2.57 , -1.16 , -0.14 ,
 3.45 , 0.05 , -1.23 , 3.33 , -0.68 , 0.88 , 
 3.76 , -1.31 , -0.76 , 2.92 , 1.36 , 0.59 ,
 3.91 , -0.39 , 0.10 , 2.91 , -0.79 , 0.31 ,
 3.53 , -0.57 , 0.48 , 3.79 , 0.59 , 1.36 , 
 5.04 , 1.60 , -0.21 , 3.64 , 1.69 , -3.26 ,
 5.66 , 1.85 , -1.45 , 3.05 , -0.70 , 0.75 , 
 3.60 , -1.27 , -1.25 , 3.59 , -0.74 , 0.43 ,
 3.93 , -0.71 , 0.41 , 3.95 , -0.21 , 0.99 , 
 4.87 , -0.36 , 2.31 , 3.63 , -0.23 , 2.57 , 
 3.43 , -0.75 , -1.94 , 3.04 , 0.22 , 0.15 , 
 3.23 , -0.34 , 5.23 , 3.32 , -1.33 , 0.52 , 
 3.38 , -1.27 , 0.40 , 3.06 , 0.01 , 0.67 , 
 4.40 , -0.14 , 0.36 , 4.24 , -1.41 , 0.79 , 
 5.91 , 1.57 , 4.86 , 6.65 , -2.39 , 2.72 , 
 4.58 , 0.05 , 0.64 , 2.66 , 0.70 , -0.75 , 
 4.71 , 0.03 , 3.77 , 5.21 , -2.24 , 4.84 , 
 5.17 , 0.28 , 4.70 , 4.80 , -2.83 , 3.38 , 
 3.52 , -0.39 , -0.35 , 2.84 , 0.67 , -0.59 , 
 3.42 , -0.83 , 0.13 , 3.25 , -2.64 , 2.55 ,
 3.91 , -0.63 , 0.83 , 3.86 , 0.17 , 0.75 , 
 5.82 , 1.70 , 1.60 , 3.72 , -0.30 , -0.51 , 
 3.92 , -0.63 , 0.32 , 3.57 , -0.22 , 0.50 , 
 5.72 , 2.74 , 2.60 , 4.32 , -2.54 , 1.75 , 
 3.43 , -0.38 , -1.41 , 2.49 , 0.46 , 0.19 ,
 3.26 , -0.04 , 0.86 , 3.66 , 0.79 , -0.04 , 
 5.32 , 0.59 , 2.11 , 3.77 , -1.00 , -0.73 , 
 5.20 , 1.67 , -0.40 , 10.21 , -3.02 , -0.30 ,
 4.32 , -0.57 , 0.51 , 2.84 , 0.69 , 0.18 , 
 4.67 , 0.62 , 2.83 , 4.04 , -2.36 , 0.65 , 
 4.38 , 0.52 , -2.66 , 2.40 , -0.23 , -1.13 , 
 5.43 , -0.79 , 3.77 , 5.00 , -1.73 , 4.90 ,
 3.50 , -1.59 , 0.83 , 3.92 , 1.00 , 1.48 , 
 3.58 , -1.88 , -0.38 , 3.56 , 0.75 , 0.12 , 
 5.39 , -0.21 , 0.47 , 4.25 , -1.07 , 0.60 , 
 4.11 , -0.72 , -0.36 , 4.22 , -0.39 , 0.32 , 
 5.15 , 0.41 , -0.02 , 5.40 , -0.76 , 0.71 , 
 3.94 , -1.33 , -5.22 , 2.16 , 2.13 , -0.95 , 
 3.53 , 0.27 , 1.22 , 2.11 , 0.07 , 0.04 ,  //45 jump data

 4.55, -1.07, 1.73, 3.50, -0.01, 0.57,
4.09, -1.70, 0.43, 4.14, 0.13, 0.53,
3.82, 0.04, 3.05, 5.03, -2.96, 0.84,
4.45, 0.08, 0.82, 4.62, -0.88, 1.38,
5.35, -0.62, 0.41, 3.86, 0.62, 0.50,
3.88, -0.00, -1.10, 2.53, 1.10, -0.49,
4.38, -1.99, 1.09, 4.91, 1.09, 0.80,
3.11, -2.12, -0.29, 3.57, 0.35, 0.49,
4.61, -0.98, 1.03, 5.10, 0.86, 0.69,
4.59, -0.50, -0.79, 4.44, 0.25, 1.48,
3.93, -1.26, 0.12, 4.70, 0.56, 0.63,
3.49, -0.50, 0.80, 3.50, -0.16, 1.79,
3.52, -0.10, 0.95, 4.97, -5.66, 3.73,
4.70, 0.88, 2.35, 4.42, -1.94, 1.66,
4.21, -1.39, 2.14, 4.55, -0.07, 1.11,
3.36, -0.36, 0.35, 2.35, 0.23, 0.41,
5.79, -0.12, -0.88, 4.34, -0.04, 0.51,
4.01, -1.82, 0.29, 5.54, 0.88, 0.68,
4.71, -0.85, -0.02, 3.46, -0.78, 3.95,
3.72, -2.27, -0.23, 3.93, -0.02, 0.23,
4.47, 1.07, -1.07, 3.85, 1.85, 2.01,
5.24, -1.66, 0.74, 5.03, 0.64, 0.84,
4.79, -2.24, 1.60, 4.28, -0.76, 1.33,
4.54, -0.87, 1.08, 5.25, -0.25, 0.89,
4.94, -1.77, 1.71, 3.48, 0.04, 0.87,
3.57, -2.02, -0.67, 2.80, 0.40, 0.18,
7.31, 1.89, 3.37, 3.30, -1.21, 1.88,
3.96, -0.13, 0.51, 2.99, -0.51, 0.58,
4.37, 0.22, 0.21, 5.25, -1.08, 3.48,
4.45, -1.18, 0.39, 5.39, 0.18, 0.30,
4.95, -1.32, 1.49, 4.36, -0.30, 1.00,
5.49, -1.61, 1.61, 4.49, -0.24, 1.45,
5.55, -2.33, 2.48, 6.50, -1.79, 1.71,
5.67, -0.66, 0.18, 5.50, -0.24, 2.39,
4.91, -0.57, 0.75, 4.48, 0.48, 0.76,
3.76, -2.61, -0.98, 2.89, 0.95, -0.68,
5.60, -1.31, 0.95, 3.73, -0.86, 1.39,
3.41, -0.76, 0.79, 5.48, -0.47, 2.51,
4.32, -1.57, 1.29, 4.49, 0.49, -0.18,
4.08, 0.86, -2.24, 3.22, 0.95, -0.50,
5.42, -2.01, 1.73, 4.23, -0.23, 2.27,
3.88, 0.27, 0.24, 3.50, 1.93, -0.05,
3.04, 1.00, -1.43, 5.63, -2.01, 2.51,
4.35, -2.54, 2.25, 2.65, 1.19, 0.64,
3.54, -0.34, -0.32, 1.95, 1.12, 1.03,// 45 open_jump_data

1.88 , -0.80 , 1.70 , 0.42 , 0.84 , 0.25 ,
1.70 , 0.30 , -0.38 , 0.25 , 1.57 , -1.10 ,
2.02 , -1.02 , 1.36 , -0.04 , 0.63 , -0.31 ,
2.61 , 0.04 , 1.43 , 0.39 , 1.13 , -0.33 ,
3.11 , 0.58 , 0.85 , 0.82 , -0.19 , -0.80 ,
3.32 , 0.02 , 0.88 , 2.23 , -1.77 , 0.98 ,
3.74 , 2.17 , 0.85 , 0.93 , -0.72 , 0.50 ,
3.64 , 0.93 , 0.45 , 0.52 , 1.82 , -1.00 ,
3.88 , 2.07 , 0.83 , 1.20 , -0.89 , 0.38 ,
2.58 , -0.15 , 0.68 , 3.59 , -0.41 , 1.02 ,
2.59 , 0.26 , -0.02 , 0.93 , -0.30 , -0.06 ,
2.87 , 0.52 , 0.37 , 4.65 , -1.38 , 3.85 ,
2.32 , 0.46 , 1.75 , 1.24 , 0.36 , -0.05 ,
2.77 , 0.96 , -0.18 , 4.61 , -1.01 , 4.45 ,
2.43 , 0.29 , 0.18 , 2.57 , -1.46 , 1.95 ,
2.45 , 0.29 , 0.71 , 2.11 , -1.27 , 0.18 ,
3.09 , 1.52 , 1.29 , 0.13 , 1.07 , -0.16 ,
3.99 , 1.57 , 1.17 , 1.13 , -0.59 , 0.38 ,
2.54 , 0.75 , 0.66 , 4.07 , -1.82 , 1.82 ,
2.45 , 0.29 , 0.71 , 2.11 , -1.27 , 0.18 ,
2.89 , -0.63 , 0.46 , 1.65 , -0.77 , 0.40 ,
2.51 , 0.23 , 0.73 , 3.91 , -1.36 , 0.42 ,
3.30 , 1.27 , 0.04 , 0.82 , -0.05 , 0.66 ,
3.02 , 0.32 , 0.49 , 2.87 , -0.12 , 1.61 ,
2.70 , 0.91 , 0.38 , 2.86 , 0.98 , 2.32 ,
2.10 , 0.95 , 0.67 , 0.50 , 0.50 , -0.30 ,
2.31 , 0.77 , 0.05 , 1.88 , 0.15 , -1.82 ,
2.21 , 1.31 , -0.27 , 0.88 , 0.82 , -1.82 ,
2.52 , 0.28 , 0.23 , 1.12 , 0.97 , 0.74 ,
2.46 , 0.61 , 0.53 , 4.78 , -1.73 , 0.64 ,
2.79 , 1.17 , 0.18 , 1.23 , -0.45 , 0.67 ,
3.07 , -0.17 , 0.47 , 1.70 , -1.97 , -0.17 ,
2.86 , 1.24 , -0.15 , 1.43 , -1.14 , 0.69 ,
2.88 , 0.73 , 0.13 , 2.60 , -0.71 , -0.45 ,
3.46 , 0.98 , 0.88 , 1.38 , -0.62 , -0.29 ,
3.36 , 1.54 , -0.63 , 1.55 , -0.71 , 0.63 ,
2.75 , 0.67 , 0.43 , 4.12 , -1.18 , 2.41 ,
4.36 , 0.71 , 2.07 , 1.70 , -0.52 , 0.11 ,
2.69 , 0.94 , 0.46 , 1.31 , -0.73 , 0.79 ,
2.57 , 1.20 , 0.34 , 2.29 , -1.08 , -0.72 ,
3.35 , 0.39 , 2.24 , 1.88 , -0.55 , 0.38 ,
2.43 , 1.00 , 0.55 , 2.72 , -1.86 , 0.39 ,
2.18 , 1.12 , 1.13 , 1.99 , -0.80 , -0.27 ,
1.91 , -0.44 , 1.16 , 0.64 , -0.22 , 0.11 ,
2.52 , 0.73 , 0.02 , 3.58 , -0.11 , 0.79 ,//45 run

 1.28 ,0.29 ,  0.98  , 1.09 , -0.21 ,  -0.01 , 
 1.14 ,0.23 ,  -0.00  , 0.56 , 0.09 ,  -0.07 , 
 1.20 ,0.28 ,  0.26  , 0.89 , -0.15 ,  0.82 , 
 1.32 ,0.32 ,  -0.12  , 1.01 , -0.02 ,  0.18 , 
 2.21 ,0.32 ,  1.66  , 1.16 , -0.51 ,  0.11 , 
 1.95 ,-0.36 ,  0.19  , 1.16 , -0.58 ,  0.27 , 
 1.45 ,0.89 ,  -0.05  , 1.23 , 0.06 ,  -0.16 , 
 1.46 ,0.13 ,  1.20  , 1.13 , -0.28 ,  0.13 , 
 1.28 ,-0.01 ,  0.09  , 1.62 , -0.20 ,  1.11 , 
 1.65 ,-0.17 ,  1.14  , 1.10 , -0.30 ,  0.14 ,
 1.23 ,0.03 ,  0.19  , 1.76 , -0.31 ,  1.33 , 
 1.30 ,0.21 ,  0.02  , 1.32 , -0.52 ,  -0.70 ,
 1.34 ,0.13 ,  1.15  , 1.15 , -0.41 ,  0.21 ,
 1.48 ,0.22 ,  1.13  , 1.11 , -0.30 ,  0.23 ,
 1.38 ,-0.26 ,  -0.10  , 1.15 , -0.54 ,  0.21 ,
 1.29 ,0.10 ,  0.26  , 0.27 , 0.20 ,  -0.05 , 
 1.29 ,-0.12 ,  -0.53  , 0.78 , 0.02 ,  -0.53 , 
 1.57 ,-0.07 ,  -0.02  , 1.20 , -0.40 ,  0.34 ,
 1.38 ,0.23 ,  -0.69  , 0.88 , -0.58 ,  -0.17 ,
 1.27 ,0.13 ,  0.18  , 2.30 , -0.71 ,  1.45 ,
 1.23 ,0.16 ,  0.49  , 1.21 , 0.12 ,  -0.66 ,
 1.46 ,0.24 ,  -0.36  , 1.17 , -0.96 ,  -0.18 ,
 1.01 ,-0.20 ,  0.45  , 1.08 , -0.41 ,  0.02 ,
 1.09 ,0.17 ,  0.11  , 0.71 , -0.13 ,  -0.08 ,
 1.31 ,0.14 ,  0.16  , 2.12 , -0.86 ,  1.42 ,
 1.51 ,0.18 ,  0.31  , 1.46 , -0.46 ,  0.13 ,
 1.51 ,0.18 ,  0.31  , 1.46 , -0.46 ,  0.13 , 
 1.29 ,-0.41 ,  -0.07  , 1.12 , -0.51 ,  0.24 , 
 1.32 ,0.34 ,  0.15  , 0.98 , 0.29 ,  -0.18 ,
 1.24 ,0.07 ,  0.07  , 1.85 , -0.48 ,  1.45 ,
 1.32 ,0.15 ,  0.43  , 1.59 , -0.16 ,  0.25 , 
 1.37 ,0.19 ,  -0.17  , 1.04 , -0.84 ,  -0.05 , 
 1.34 ,0.24 ,  0.46  , 1.25 , 0.27 ,  -0.62 ,
 1.45 ,0.14 ,  -0.49  , 0.97 , -0.79 ,  -0.25 , 
 1.32 ,0.23 ,  0.40  , 1.52 , 0.09 ,  -0.27 , 
 1.32 ,0.02 ,  -0.25  , 1.08 , -0.86 ,  0.08 ,  
 1.32 ,0.29 ,  0.33  , 1.45 , 0.11 ,  -0.46 ,
 1.34 ,-0.13 ,  0.00  , 1.06 , -0.85 ,  -0.18 ,
 1.32 ,0.39 ,  0.29  , 1.14 , 0.45 ,  -0.75 ,
 1.32 ,0.09 ,  -0.24  , 1.01 , -0.84 ,  -0.01 ,
 1.34 ,0.20 ,  0.32  , 0.52 , 0.21 ,  -0.80 , 
 1.55 ,-0.25 ,  0.10  , 1.09 , -0.62 ,  0.22 ,  
 1.40 ,-0.09 ,  -0.98  , 0.57 , 0.17 ,  -0.72 , 
 1.29 ,0.36 ,  0.25  , 1.36 , 0.03 ,  -0.87 , 
 1.43 ,-0.01 ,  -0.63  , 1.31 , -0.38 ,  0.12 ,  //45 walk



};	    // You can put your train dataset here
int train_data_output[train_data_num ][target_num] = {
    1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
1,0,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,1,0,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,1,0,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
0,0,0,1,
};      // Label of the train data

// Create test dataset/output
float test_data_input[test_data_num][input_length] = {
	
    

};		// You can put your test dataset here

int test_data_output[test_data_num][target_num] = {
 

};		// Label of the test data

int ReportEvery10;
int RandomizedIndex[train_data_num];
long  TrainingCycle;
float Rando;
float Error;
float Accum;

float data_mean[6] ={0};
float data_std[6] ={0};

float Hidden_1[HiddenNodes_1];
float Hidden_2[HiddenNodes_2];
float Output[target_num];

float HiddenWeights_1[input_length+1][HiddenNodes_1];
float HiddenWeights_2[HiddenNodes_1+1][HiddenNodes_2];
float OutputWeights[HiddenNodes_2+1][target_num];

float HiddenDelta_1[HiddenNodes_1];
float HiddenDelta_2[HiddenNodes_2];

float OutputDelta[target_num];
float ChangeHiddenWeights_1[input_length+1][HiddenNodes_1];
float ChangeHiddenWeights_2[HiddenNodes_1+1][HiddenNodes_2];
float ChangeOutputWeights[HiddenNodes_2+1][target_num];

int target_value;
int out_value;
int max;

void SYS_Init(void)
{
    SYS_UnlockReg();
    CLK_EnableXtalRC(CLK_PWRCTL_HIRCEN_Msk);
    CLK_WaitClockReady(CLK_STATUS_HIRCSTB_Msk);
    CLK_SetHCLK(CLK_CLKSEL0_HCLKSEL_HIRC, CLK_CLKDIV0_HCLK(1));
    CLK->PCLKDIV = (CLK_PCLKDIV_APB0DIV_DIV2 | CLK_PCLKDIV_APB1DIV_DIV2);
    CLK_SetModuleClock(UART0_MODULE, CLK_CLKSEL1_UART0SEL_HIRC, CLK_CLKDIV0_UART0(1));
    CLK_EnableModuleClock(UART0_MODULE);
    CLK_EnableModuleClock(ADC_MODULE);
    CLK_SetModuleClock(ADC_MODULE, CLK_CLKSEL2_ADCSEL_PCLK1, CLK_CLKDIV0_ADC(1));
    SystemCoreClockUpdate();
    SYS->GPB_MFPH = (SYS->GPB_MFPH & ~(SYS_GPB_MFPH_PB12MFP_Msk | SYS_GPB_MFPH_PB13MFP_Msk)) |
                    (SYS_GPB_MFPH_PB12MFP_UART0_RXD | SYS_GPB_MFPH_PB13MFP_UART0_TXD);
    GPIO_SetMode(PB, BIT0 | BIT1 | BIT2, GPIO_MODE_INPUT);
    SYS->GPB_MFPL = (SYS->GPB_MFPL & ~(SYS_GPB_MFPL_PB0MFP_Msk | SYS_GPB_MFPL_PB1MFP_Msk | SYS_GPB_MFPL_PB2MFP_Msk)) |
                    (SYS_GPB_MFPL_PB0MFP_ADC0_CH0 | SYS_GPB_MFPL_PB1MFP_ADC0_CH1 | SYS_GPB_MFPL_PB2MFP_ADC0_CH2);
    GPIO_DISABLE_DIGITAL_PATH(PB, BIT0 | BIT1 | BIT2);
    SYS_LockReg();
}

void ADC_IRQHandler(void)
{
    g_u32AdcIntFlag = 1;
    ADC_CLR_INT_FLAG(ADC, ADC_ADF_INT);
}

void UART0_Init(void)
{
    SYS_ResetModule(UART0_RST);
    UART_Open(UART0, 115200);
}

void scale_data()
{
		float sum[input_length] = {0};
		int i, j;

		// Compute Data Mean
		for(i = 0; i < train_data_num; i++){
			for(j = 0; j < input_length; j++){
				sum[j] += train_data_input[i][j];
			}
		}
		for(j = 0; j < input_length ; j++){
			data_mean[j] = sum[j] / train_data_num;
			printf("MEAN: %.2f\n", data_mean[j]);
			sum[j] = 0.0;
		}

		// Compute Data STD
		for(i = 0; i < train_data_num; i++){
			for(j = 0; j < input_length ; j++){
				sum[j] += pow(train_data_input[i][j] - data_mean[j], 2);
			}
		}
		for(j = 0; j < input_length; j++){
			data_std[j] = sqrt(sum[j]/train_data_num);
			printf("STD: %.2f\n", data_std[j]);
			sum[j] = 0.0;
		}
}

void normalize(float *data)
{
		int i;

		for(i = 0; i < input_length; i++){
			data[i] = (data[i] - data_mean[i]) / data_std[i];
		}
}

int train_preprocess()
{
    int i;

    for(i = 0 ; i < train_data_num ; i++)
    {
        normalize(train_data_input[i]);
    }

    return 0;
}

int test_preprocess()
{
    int i;

    for(i = 0 ; i < test_data_num ; i++)
    {
        normalize(test_data_input[i]);
    }

    return 0;
}




int data_setup()
{
    int i;
		//int j;
		int p, ret;
	    uint32_t u32ChannelCount;
    int32_t i32ConversionData[3];
		unsigned int seed = 1;
    printf("ADC started...\n");
	ADC_POWER_ON(ADC);
				ADC_Open(ADC, ADC_ADCR_DIFFEN_SINGLE_END, ADC_ADCR_ADMD_SINGLE_CYCLE, BIT0 | BIT1 | BIT2);
        ADC_CLR_INT_FLAG(ADC, ADC_ADF_INT);
        ADC_ENABLE_INT(ADC, ADC_ADF_INT);
        NVIC_EnableIRQ(ADC_IRQn);
        g_u32AdcIntFlag = 0;
        ADC_START_CONV(ADC);
        while (g_u32AdcIntFlag == 0);
        ADC_DISABLE_INT(ADC, ADC_ADF_INT);
        for (u32ChannelCount = 0; u32ChannelCount < 3; u32ChannelCount++)
        {
            i32ConversionData[u32ChannelCount] = ADC_GET_CONVERSION_DATA(ADC, u32ChannelCount);
					seed *=i32ConversionData[u32ChannelCount];
        }
        printf("{%d, %d, %d}\n", i32ConversionData[0], i32ConversionData[1], i32ConversionData[2]);
        CLK_SysTickDelay(500000);

		printf("ADC conversion done!\n");
		for(i = 0; i < 3; i++)
    {
				seed *= ADC_GET_CONVERSION_DATA(ADC, i);
    }
		seed *= 1000;
		printf("\nRandom seed: %d\n", seed);
    srand(seed);

    ReportEvery10 = 1;
    for( p = 0 ; p < train_data_num ; p++ )
    {
        RandomizedIndex[p] = p ;
    }

	scale_data();
    ret = train_preprocess();
    ret |= test_preprocess();
    if(ret) //Error Check
        return 1;

    return 0;
}


void run_train_data()
{
    int i, j, p;
    int correct=0;
    float accuracy = 0;
    printf("Train result:\n");
    for( p = 0 ; p < train_data_num ; p++ )
    {
        max = 0;
        for (i = 1; i < target_num; i++)
        {
            if (train_data_output[p][i] > train_data_output[p][max]) {
                max = i;
            }
        }
        target_value = max;

    /******************************************************************
    * Compute hidden layer activations
    ******************************************************************/

        for( i = 0 ; i < HiddenNodes_1 ; i++ ) {
            Accum = HiddenWeights_1[input_length][i] ;
            for( j = 0 ; j < input_length ; j++ ) {
                Accum += train_data_input[p][j] * HiddenWeights_1[j][i] ;
            }
            Hidden_1[i] = 1.0/(1.0 + exp(-Accum)) ;
        }
        for( i = 0 ; i < HiddenNodes_2 ; i++ ) {
            Accum = HiddenWeights_2[HiddenNodes_1][i] ;
            for( j = 0 ; j < HiddenNodes_1; j++ ) {
                Accum += Hidden_1[j] * HiddenWeights_2[j][i] ;
            }
            Hidden_2[i] = 1.0/(1.0 + exp(-Accum)) ;
        }

    /******************************************************************
    * Compute output layer activations and calculate errors
    ******************************************************************/

        for( i = 0 ; i < target_num ; i++ ) {
            Accum = OutputWeights[HiddenNodes_2][i] ;
            for( j = 0 ; j < HiddenNodes_2 ; j++ ) {
                Accum += Hidden_2[j] * OutputWeights[j][i] ;
            }
            Output[i] = 1.0/(1.0 + exp(-Accum)) ;
        }

        max = 0;
        for (i = 1; i < target_num; i++)
        {
            if (Output[i] > Output[max]) {
                max = i;
            }
        }
        out_value = max;

        if(out_value!=target_value)
            printf("Error --> Training Pattern: %d,Target : %d, Output : %d\n", p, target_value, out_value);
        else
            correct++;
        }


        // Calculate accuracy
        accuracy = (float)correct / train_data_num;
        printf ("Accuracy1 = %.2f /100 \n",accuracy*100);

}
void run_test_data()
{
    int i, j, p;
    int correct=0;
    float accuracy = 0;
    printf("Test result:\n");
    for( p = 0 ; p < test_data_num ; p++ )
    {
        max = 0;
        for (i = 1; i < target_num; i++)
        {
            if (test_data_output[p][i] > test_data_output[p][max]) {
                max = i;
            }
        }
        target_value = max;

    /******************************************************************
    * Compute hidden layer activations
    ******************************************************************/

        for( i = 0 ; i < HiddenNodes_1 ; i++ ) {
            Accum = HiddenWeights_1[input_length][i] ;
            for( j = 0 ; j < input_length ; j++ ) {
                Accum += test_data_input[p][j] * HiddenWeights_1[j][i] ;
            }
            Hidden_1[i] = 1.0/(1.0 + exp(-Accum)) ;
        }
        for( i = 0 ; i < HiddenNodes_2 ; i++ ) {
            Accum = HiddenWeights_2[HiddenNodes_1][i] ;
            for( j = 0 ; j < HiddenNodes_1 ; j++ ) {
                Accum += Hidden_1[j] * HiddenWeights_2[j][i] ;
            }
            Hidden_2[i] = 1.0/(1.0 + exp(-Accum)) ;
        }

    /******************************************************************
    * Compute output layer activations and calculate errors
    ******************************************************************/

        for( i = 0 ; i < target_num ; i++ ) {
            Accum = OutputWeights[HiddenNodes_2][i] ;
            for( j = 0 ; j < HiddenNodes_2 ; j++ ) {
                Accum += Hidden_2[j] * OutputWeights[j][i] ;
            }
            Output[i] = 1.0/(1.0 + exp(-Accum)) ;
        }
        max = 0;
        for (i = 1; i < target_num; i++)
        {
            if (Output[i] > Output[max]) {
                max = i;
            }
        }
        out_value = max;

        if(out_value!=target_value)
            printf("Error --> Training Pattern: %d,Target : %d, Output : %d\n", p, target_value, out_value);
        else
            correct++;
        }
        // Calculate accuracy
        accuracy = (float)correct / test_data_num;

        printf ("Accuracy2 = %.2f /100 \n",accuracy*100);
}

float Get_Train_Accuracy()
{
    int i, j, p;
    int correct = 0;
		float accuracy = 0;
    for (p = 0; p < train_data_num; p++)
    {
/******************************************************************
* Compute hidden layer activations
******************************************************************/

        for( i = 0 ; i < HiddenNodes_1 ; i++ ) {
            Accum = HiddenWeights_1[input_length][i] ;
            for( j = 0 ; j < input_length ; j++ ) {
                Accum += train_data_input[p][j] * HiddenWeights_1[j][i] ;
            }
            Hidden_1[i] = 1.0/(1.0 + exp(-Accum)) ;
        }
        for( i = 0 ; i < HiddenNodes_2 ; i++ ) {
            Accum = HiddenWeights_2[HiddenNodes_1][i] ;
            for( j = 0 ; j < HiddenNodes_1 ; j++ ) {
                Accum += Hidden_1[j] * HiddenWeights_2[j][i] ;
            }
            Hidden_2[i] = 1.0/(1.0 + exp(-Accum)) ;
        }
/******************************************************************
* Compute output layer activations
******************************************************************/

        for( i = 0 ; i < target_num ; i++ ) {
            Accum = OutputWeights[HiddenNodes_2][i] ;
            for( j = 0 ; j < HiddenNodes_2 ; j++ ) {
                Accum += Hidden_2[j] * OutputWeights[j][i] ;
            }
            Output[i] = 1.0/(1.0 + exp(-Accum)) ;
        }


        //get target value
        max = 0;
        for (i = 1; i < target_num; i++)
        {
            if (train_data_output[p][i] > train_data_output[p][max]) {
                max = i;
            }
        }
        target_value = max;
        //get output value
        max = 0;
        for (i = 1; i < target_num; i++)
        {
            if (Output[i] > Output[max]) {
                max = i;
            }
        }
        out_value = max;
        //compare output and target
        if (out_value==target_value)
        {
            correct++;
        }
    }

    // Calculate accuracy
    accuracy = (float)correct / train_data_num;
		//printf("correct %d\n",correct);
    return accuracy;
}

void load_weight()
{
    int i, j;

    // Input → Hidden-1
    printf("\n======= HiddenWeights_1 (input→Hidden1) =======\n");
    printf("{\n");
    for (i = 0; i <= input_length; i++) {
        printf("  {");
        for (j = 0; j < HiddenNodes_1; j++) {
            printf("%f%s", HiddenWeights_1[i][j],
                   (j < HiddenNodes_1 - 1) ? ", " : "");
        }
        printf("}%s\n", (i < input_length) ? "," : "");
    }
    printf("}\n");

    // Hidden-1 → Hidden-2
    printf("\n======= HiddenWeights_2 (Hidden1→Hidden2) =======\n");
    printf("{\n");
    for (i = 0; i <= HiddenNodes_1; i++) {
        printf("  {");
        for (j = 0; j < HiddenNodes_2; j++) {
            printf("%f%s", HiddenWeights_2[i][j],
                   (j < HiddenNodes_2 - 1) ? ", " : "");
        }
        printf("}%s\n", (i < HiddenNodes_1) ? "," : "");
    }
    printf("}\n");

    // Hidden-2 → Output
    printf("\n======= OutputWeights (Hidden2→output) =======\n");
    printf("{\n");
    for (i = 0; i <= HiddenNodes_2; i++) {
        printf("  {");
        for (j = 0; j < target_num; j++) {
            printf("%f%s", OutputWeights[i][j],
                   (j < target_num - 1) ? ", " : "");
        }
        printf("}%s\n", (i < HiddenNodes_2) ? "," : "");
    }
    printf("}\n");
}

void AdcSingleCycleScanModeTest()
{
		int i, j;
    uint32_t u32ChannelCount;
    float single_data_input[6];
		char output_string[10] = {NULL};

    printf("\n");
		printf("[Phase 3] Start Prediction ...\n\n");
		PB2=1;
    while(1)
    {

				/* Set the ADC operation mode as single-cycle, input mode as single-end and
                 enable the analog input channel 0, 1, 2 and 3 */
        ADC_Open(ADC, ADC_ADCR_DIFFEN_SINGLE_END, ADC_ADCR_ADMD_SINGLE_CYCLE, 0x7);

        /* Power on ADC module */
        ADC_POWER_ON(ADC);

        /* Clear the A/D interrupt flag for safe */
        ADC_CLR_INT_FLAG(ADC, ADC_ADF_INT);

        /* Start A/D conversion */
        ADC_START_CONV(ADC);

        /* Wait conversion done */
        while(!ADC_GET_INT_FLAG(ADC, ADC_ADF_INT));

        for(u32ChannelCount = 0; u32ChannelCount < 3; u32ChannelCount++)
        {
            single_data_input[u32ChannelCount] = ADC_GET_CONVERSION_DATA(ADC, u32ChannelCount);
        }
				normalize(single_data_input);


				// Compute hidden layer activations
				for( i = 0 ; i < HiddenNodes_1 ; i++ ) {
						Accum = HiddenWeights_1[input_length][i] ;
						for( j = 0 ; j < input_length ; j++ ) {
								Accum += single_data_input[j] * HiddenWeights_1[j][i] ;
						}
						Hidden_1[i] = 1.0/(1.0 + exp(-Accum)) ;
				}
                for( i = 0 ; i < HiddenNodes_2 ; i++ ) {
                    Accum = HiddenWeights_2[HiddenNodes_1][i];          // bias term
                    for( j = 0 ; j < HiddenNodes_1 ; j++ ) {
                        Accum += Hidden_1[j] * HiddenWeights_2[j][i];   // feed in Hidden_1, not single_data_input
                    }
                    Hidden_2[i] = 1.0/(1.0 + exp(-Accum));
                }

				// Compute output layer activations
				for( i = 0 ; i < target_num ; i++ ) {
						Accum = OutputWeights[HiddenNodes_2][i] ;
						for( j = 0 ; j < HiddenNodes_2 ; j++ ) {
								Accum += Hidden_2[j] * OutputWeights[j][i] ;
						}
						Output[i] = 1.0/(1.0 + exp(-Accum)) ;
				}

				max = 0;
				for (i = 1; i < target_num; i++)
				{
						if (Output[i] > Output[max]) {
								max = i;
						}
				}
				out_value = max;

				switch(out_value){
						case 0:
								strcpy(output_string, "jump");
								break;
						case 1:
								strcpy(output_string, "open_close_jump");
								break;
						case 2:
								strcpy(output_string, "run");
								break;
						case 3:
								strcpy(output_string, "walk");
								break;
				}

				printf("\rPrediction output: %-8s", output_string);
				CLK_SysTickDelay(500000);


    }
}



int32_t main(void)
{
		int i, j, p, q, r;
    float accuracy=0;
	  uint32_t u32ChannelCount;
    int32_t i32ConversionData[3];
    SYS_Init();
    UART0_Init();
		GPIO_SetMode(PA, BIT2, GPIO_MODE_OUTPUT);
		PA2=0;
	  printf("\n+-----------------------------------------------------------------------+\n");
    printf("|                        00LAB8 - Machine Learning                        |\n");
    printf("+-----------------------------------------------------------------------+\n");

	    printf("\n[Phase 1] Initialize DataSet ...");
	  /* Data Init (Input / Output Preprocess) */
		if(data_setup()){
        printf("[Error] Datasets Setup Error\n");
        return 0;
    }else
				printf("Done!\n\n");

		printf("[Phase 2] Start Model Training ...\n");
		// Initialize HiddenWeights and ChangeHiddenWeights
    for( i = 0 ; i < HiddenNodes_1 ; i++ ) {
        for( j = 0 ; j <= input_length ; j++ ) {
            ChangeHiddenWeights_1[j][i] = 0.0 ;
            Rando = (float)((rand() % 100))/100;
            HiddenWeights_1[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
        }
    }
    for( i = 0 ; i < HiddenNodes_2 ; i++ ) {
        for( j = 0 ; j <= HiddenNodes_1 ; j++ ) {
            ChangeHiddenWeights_2[j][i] = 0.0 ;
            Rando = (float)((rand() % 100))/100;
            HiddenWeights_2[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
        }
    }

    // Initialize OutputWeights and ChangeOutputWeights
    for( i = 0 ; i < target_num ; i ++ ) {
        for( j = 0 ; j <= HiddenNodes_2 ; j++ ) {
            ChangeOutputWeights[j][i] = 0.0 ;
            Rando = (float)((rand() % 100))/100;
            OutputWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
        }
    }

    // Begin training
    for(TrainingCycle = 1 ; TrainingCycle < 2147483647 ; TrainingCycle++)
    {
        Error = 0.0 ;

        // Randomize order of training patterns
        for( p = 0 ; p < train_data_num ; p++) {
            q = rand()%train_data_num;
            r = RandomizedIndex[p] ;
            RandomizedIndex[p] = RandomizedIndex[q] ;
            RandomizedIndex[q] = r ;
        }

        // Cycle through each training pattern in the randomized order
        for( q = 0 ; q < train_data_num ; q++ )
        {
            p = RandomizedIndex[q];

            /* --------------------------------------------------------- */
            /*  Forward pass                                            */
            /* --------------------------------------------------------- */

            /* Compute Hidden-1 activations */
            for( i = 0 ; i < HiddenNodes_1 ; i++ ) {
                Accum = HiddenWeights_1[input_length][i] ;
                for( j = 0 ; j < input_length ; j++ ) {
                    Accum += train_data_input[p][j] * HiddenWeights_1[j][i] ;
                }
                Hidden_1[i] = 1.0/(1.0 + exp(-Accum)) ;
            }

            /* Compute Hidden-2 activations */
            for( i = 0 ; i < HiddenNodes_2 ; i++ ) {
                Accum = HiddenWeights_2[HiddenNodes_1][i] ;
                for( j = 0 ; j < HiddenNodes_1 ; j++ ) {
                    Accum += Hidden_1[j] * HiddenWeights_2[j][i] ;
                }
                Hidden_2[i] = 1.0/(1.0 + exp(-Accum)) ;
            }

            /* Compute Output layer activations and deltas */
            for( i = 0 ; i < target_num ; i++ ) {
                Accum = OutputWeights[HiddenNodes_2][i] ;
                for( j = 0 ; j < HiddenNodes_2 ; j++ ) {
                    Accum += Hidden_2[j] * OutputWeights[j][i] ;
                }
                Output[i] = 1.0/(1.0 + exp(-Accum)) ;
                OutputDelta[i] = (train_data_output[p][i] - Output[i]) * Output[i] * (1.0 - Output[i]) ;
                Error += 0.5 * (train_data_output[p][i] - Output[i]) * (train_data_output[p][i] - Output[i]) ;
            }

            /* --------------------------------------------------------- */
            /*  Back-propagation                                        */
            /* --------------------------------------------------------- */

            /* Hidden-2 deltas */
            for( i = 0 ; i < HiddenNodes_2 ; i++ ) {
                Accum = 0.0 ;
                for( j = 0 ; j < target_num ; j++ ) {
                    Accum += OutputWeights[i][j] * OutputDelta[j] ;
                }
                HiddenDelta_2[i] = Accum * Hidden_2[i] * (1.0 - Hidden_2[i]) ;
            }

            /* Hidden-1 deltas */
            for( i = 0 ; i < HiddenNodes_1 ; i++ ) {
                Accum = 0.0 ;
                for( j = 0 ; j < HiddenNodes_2 ; j++ ) {
                    Accum += HiddenWeights_2[i][j] * HiddenDelta_2[j] ;
                }
                HiddenDelta_1[i] = Accum * Hidden_1[i] * (1.0 - Hidden_1[i]) ;
            }

            /* --------------------------------------------------------- */
            /*  Weight updates (Momentum + SGD)                         */
            /* --------------------------------------------------------- */

            /* Input  -> Hidden-1 */
            for( i = 0 ; i < HiddenNodes_1 ; i++ ) {
                ChangeHiddenWeights_1[input_length][i] = LearningRate * HiddenDelta_1[i] + Momentum * ChangeHiddenWeights_1[input_length][i] ;
                HiddenWeights_1[input_length][i] += ChangeHiddenWeights_1[input_length][i] ;
                for( j = 0 ; j < input_length ; j++ ) {
                    ChangeHiddenWeights_1[j][i] = LearningRate * train_data_input[p][j] * HiddenDelta_1[i] + Momentum * ChangeHiddenWeights_1[j][i] ;
                    HiddenWeights_1[j][i] += ChangeHiddenWeights_1[j][i] ;
                }
            }

            /* Hidden-1 -> Hidden-2 */
            for( i = 0 ; i < HiddenNodes_2 ; i++ ) {
                ChangeHiddenWeights_2[HiddenNodes_1][i] = LearningRate * HiddenDelta_2[i] + Momentum * ChangeHiddenWeights_2[HiddenNodes_1][i] ;
                HiddenWeights_2[HiddenNodes_1][i] += ChangeHiddenWeights_2[HiddenNodes_1][i] ;
                for( j = 0 ; j < HiddenNodes_1 ; j++ ) {
                    ChangeHiddenWeights_2[j][i] = LearningRate * Hidden_1[j] * HiddenDelta_2[i] + Momentum * ChangeHiddenWeights_2[j][i] ;
                    HiddenWeights_2[j][i] += ChangeHiddenWeights_2[j][i] ;
                }
            }

            /* Hidden-2 -> Output */
            for( i = 0 ; i < target_num ; i++ ) {
                ChangeOutputWeights[HiddenNodes_2][i] = LearningRate * OutputDelta[i] + Momentum * ChangeOutputWeights[HiddenNodes_2][i] ;
                OutputWeights[HiddenNodes_2][i] += ChangeOutputWeights[HiddenNodes_2][i] ;
                for( j = 0 ; j < HiddenNodes_2 ; j++ ) {
                    ChangeOutputWeights[j][i] = LearningRate * Hidden_2[j] * OutputDelta[i] + Momentum * ChangeOutputWeights[j][i] ;
                    OutputWeights[j][i] += ChangeOutputWeights[j][i] ;
                }
            }
        }
        accuracy = Get_Train_Accuracy();

        // Every 10 cycles send data to terminal for display
        ReportEvery10 = ReportEvery10 - 1;
        if (ReportEvery10 == 0)
        {

            printf ("\nTrainingCycle: %ld\n",TrainingCycle);
            printf ("Error = %.5f\n",Error);
            printf ("Accuracy3 = %.2f /100 \n",accuracy*100);
            load_weight();
            //run_train_data();

            if (TrainingCycle==1)
            {
                ReportEvery10 = 9;
            }
            else
            {
                ReportEvery10 = 10;
            }
        }

        // If error rate is less than pre-determined threshold then end
        if( accuracy >= goal_acc ) break ;
    }

    printf ("\nTrainingCycle: %ld\n",TrainingCycle);
    printf ("Error = %.5f\n",Error);
    run_train_data();
    printf ("Training Set Solved!\n");
    printf ("--------\n");
    printf ("Testing Start!\n ");
    run_test_data();
    printf ("--------\n");
    ReportEvery10 = 1;
    load_weight();

		printf("\nModel Training Phase has ended.\n");

    /* Start prediction */
    AdcSingleCycleScanModeTest();

    while(1);
	
}
