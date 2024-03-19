#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>
#include <utee_defines.h>
#include <hello_world_ta.h>

#define TIME_DECLARE TEE_Time start, end, time; \
                                uint32_t delta
     
#define TIME_START TEE_GetSystemTime(&start) 
#define TIME_END	TEE_GetSystemTime(&end);	\
     	                TEE_TIME_SUB(end, start, time);	\
                        delta = time.seconds * 1000 + time.millis;      \
	    	        EMSG("it takes %d ms\n", delta)


#define MEASURE_SQRT(func1, msg)	EMSG("%d %s operations takes ", UNITS*ROUNDS, msg); \
								TIME_START;	\
								for(i = 0; i < ROUNDS; i++) {	\
									for (j = 0; j < UNITS; j++) { \
										y_arr[j] = func1(x_arr[j]);	\
									}	\
								}	\
								TIME_END

#define CHECK_SQRT(msg)    EMSG("check %s:",msg);  \
	                    for(i = 0; i < UNITS; i++){             \
                            delta_arr[i] = y_arr[i] - py_sqrt_x_arr[i];        \
                            EMSG("%f ", delta_arr[i]);        \
                        }                                       \
                        EMSG("\n\n")

#define UNITS		20
#define ROUNDS		10000

#define SQRT_X  {7.7454134134326855, 4.929154069052465, 6.069640646179461, 9.841134821231753, 1.08555314242884, 9.84442910234527, 5.737083280882002, 7.671134875387268, 6.4559455397123555, 6.03347990048145, 2.5727065326253573, 7.925506697816573, 6.928859816913366, 4.851134953805458, 6.453188194755136, 3.552161190614317, 9.534689730668246, 8.780242010644505, 5.548812842384746, 1.070885526525237}
#define SQRT_Y {2.783058284232058, 2.2201698288762652, 2.463664069263393, 3.1370583069544233, 1.041898815830424, 3.137583321976529, 2.3952209252764143, 2.769681367122808, 2.5408552772073336, 2.4563142918774563, 1.6039658763905662, 2.815227645824858, 2.632272747439628, 2.2025292174691935, 2.540312617524689, 1.8847178013204835, 3.0878292910503076, 2.9631473150426566, 2.355591824231173, 1.0348359901574922}
