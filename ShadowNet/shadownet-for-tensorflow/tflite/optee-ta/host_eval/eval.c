#include <stdio.h>
#include <math.h>
#include <time.h>

#include "eval.h"

void eval_sqrt() {
    float x_arr[UNITS] = SQRT_X; 
    float py_sqrt_x_arr[UNITS] = SQRT_Y; 
    float y_arr[UNITS]; 
    float neon_y_arr[UNITS]; 
    float delta_arr[UNITS]; 
    int i, j;

    TIME_DECLARE;

    // Baseline std::sqrt
    MEASURE_SQRT(sqrt, "sqrt");
    CHECK_SQRT("sqrt");

    // sqrtf
    MEASURE_SQRT(sqrtf, "sqrtf");
    CHECK_SQRT("sqrtf");

    // ta_sqrt
    MEASURE_SQRT(ta_sqrt, "ta_sqrt");
    CHECK_SQRT("ta_sqrt");

    // ieee754_sqrtf
    MEASURE_SQRT(ieee754_sqrtf, "ieee754_sqrtf");
    CHECK_SQRT("ieee754_sqrtf");

    // neon_sqrt
    MEASURE_NEON_SQRT("neon_sqrtf");
    CHECK_SQRT("neon_sqrtf");

}

void eval_add() {
    float x_arr[UNITS] = ADD_X; 
    float y_arr[UNITS] = ADD_Y; 
    float py_add_arr[UNITS] = ADD_Z; 
    float z_arr[UNITS]; 
    float delta_arr[UNITS]; 
    int i;

    TIME_DECLARE;

    // normal add 
    MEASURE_ADD(normal_array_add, "normal_array_add");
    CHECK_ADD("normal_array_add");

    // neon add 
    MEASURE_ADD(neon_array_add, "neon_array_add");
    CHECK_ADD("neon_array_add");
}

void eval_muladd() {
    float x_arr[UNITS] = MULADD_X; 
    float y_arr[UNITS] = MULADD_Y; 
    float s_arr[UNITS] = MULADD_S; 
    float py_muladd_arr[UNITS] = MULADD_Z; 
    float z_arr[UNITS]; 
    float delta_arr[UNITS]; 
    int i;

    TIME_DECLARE;

    // normal muladd 
    MEASURE_MULADD(normal_muladd, "normal_muladd");
    CHECK_MULADD("normal_muladd");

    // neon muladd 
    MEASURE_MULADD(neon_muladd, "neon_muladd");
    CHECK_MULADD("neon_muladd");
}

int main(){
    eval_sqrt();
    eval_add();
    eval_muladd();
    return 0;
}
