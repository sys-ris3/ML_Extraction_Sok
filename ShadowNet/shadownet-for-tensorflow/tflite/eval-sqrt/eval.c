#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sqrt.h>

#include "eval.h"

int main(){
    int i, j, round;
    float x_arr[UNITS] = SQRT_X; 
    float py_sqrt_x_arr[UNITS] = SQRT_Y; 
    float y_arr[UNITS]; 
    float delta_arr[UNITS]; 

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

    return 0;
}
