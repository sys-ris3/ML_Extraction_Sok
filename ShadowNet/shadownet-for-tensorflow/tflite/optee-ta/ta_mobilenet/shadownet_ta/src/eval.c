#include <math_ta.h>
#include "eval.h"

int eval(){
    int i, j, round;
    float x_arr[UNITS] = SQRT_X; 
    float py_sqrt_x_arr[UNITS] = SQRT_Y; 
    float y_arr[UNITS]; 
    float delta_arr[UNITS]; 

    TIME_DECLARE;

    // sqrtf
    MEASURE_SQRT(sqrtf, "sqrtf");
    CHECK_SQRT("sqrtf");

    // ta_sqrt
    MEASURE_SQRT(ta_sqrt, "ta_sqrt");
    CHECK_SQRT("ta_sqrt");

    return 0;
}
