#define UNITS	        32
#define ROUNDS		100000

#define TIME_DECLARE clock_t start, end;	\
     				double cpu_time_used
     
#define TIME_START start = clock()
#define TIME_END	end = clock();		\
     	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000.0;	\
		printf(" %lf ms\n", cpu_time_used)


#define MEASURE_SQRT(func1, msg)	printf("%d %s operations takes ", UNITS*ROUNDS, msg); \
								TIME_START;	\
								for(i = 0; i < ROUNDS; i++) {	\
									for (j = 0; j < UNITS; j++) { \
										y_arr[j] = func1(x_arr[j]);	\
									}	\
								}	\
								TIME_END

#define MEASURE_NEON_SQRT(msg)	printf("%d %s operations takes ", UNITS*ROUNDS, msg); \
								TIME_START;	\
								for(i = 0; i < ROUNDS; i++) {	\
                                                                        sqrt_buf((float*)x_arr, (float *)neon_y_arr, UNITS);    \
								}	\
								TIME_END

#define CHECK_SQRT(msg)    printf("check %s:",msg);  \
	                    for(i = 0; i < UNITS; i++){             \
                            delta_arr[i] = y_arr[i] - py_sqrt_x_arr[i];        \
                            printf("%f,%f,%f ", y_arr[i], py_sqrt_x_arr[i], delta_arr[i]);        \
                        }                                       \
                        printf("\n\n")

#define MEASURE_ADD(add_func, msg)       printf("%d %s operations takes ", UNITS*ROUNDS, msg); \
								TIME_START;	\
								for(i = 0; i < ROUNDS; i++) {	\
                                                                        add_func(x_arr, y_arr, z_arr, UNITS);       \
								}	\
								TIME_END
#define CHECK_ADD(msg)    printf("check %s:",msg);  \
	                    for(i = 0; i < UNITS; i++){             \
                            delta_arr[i] = z_arr[i] - py_add_arr[i];        \
                            printf("%f,%f,%f ", delta_arr[i], z_arr[i], py_add_arr[i]);        \
                        }                                       \
                        printf("\n\n")

#define MEASURE_MULADD(muladd_func, msg)       printf("%d %s operations takes ", UNITS*ROUNDS, msg); \
								TIME_START;	\
								for(i = 0; i < ROUNDS; i++) {	\
                                                                        muladd_func(x_arr, s_arr, y_arr, z_arr, UNITS);       \
								}	\
								TIME_END
#define CHECK_MULADD(msg)    printf("check %s:",msg);  \
	                    for(i = 0; i < UNITS; i++){             \
                            delta_arr[i] = z_arr[i] - py_muladd_arr[i];        \
                            printf("%f,%f,%f ", delta_arr[i], z_arr[i], py_muladd_arr[i]);        \
                        }                                       \
                        printf("\n\n")

#define SQRT_X  {8.12834280837455, 1.8694902273752112, 7.100721528340104, 5.0715969160207806, 8.28127089949316, 1.0005955419468393, 0.8519002757414462, 2.745164643132636, 8.932923171869673, 7.7600854326383795, 4.429125496113688, 5.47700802114878, 3.6693358480425275, 7.215793253684671, 1.6836797086704658, 0.12582375125029577, 3.0662518637972136, 3.920993971556043, 9.217976118282005, 2.668533322150126, 2.560566951074107, 9.673840335057937, 9.394588669232624, 1.3429126577982375, 9.141576213663098, 4.759967214069767, 1.722280551676374, 4.324609788087597, 7.477770270291639, 6.043288893555349, 6.060451984623195, 3.388529339910993}
#define SQRT_Y {2.851024869827436, 1.3672930290816272, 2.664717907835669, 2.252020629572647, 2.8777197395669303, 1.0002977266528397, 0.922984439598765, 1.6568538387958776, 2.988799620561685, 2.785692989659553, 2.1045487630638755, 2.3403008398812277, 1.915551055973849, 2.6862228600182583, 1.2975668416965909, 0.35471643780673, 1.751071632971425, 1.9801499871363388, 3.0361120068735943, 1.6335646060533162, 1.600177162402372, 3.1102797840480423, 3.065059325564943, 1.158841083927489, 3.0235039628985274, 2.1817349092109626, 1.312356869024723, 2.0795696160714594, 2.7345512008904933, 2.4583101703315124, 2.4617985264077147, 1.8407958441693073}

#define ADD_X {7.7813380289658705, 1.8000850149193126, 2.7487139975987294, 5.7666100928883175, 8.404442374000572, 2.0217607159920203, 6.561487140941679, 4.321629444398193, 1.2656368346214752, 6.185995206874719, 5.011585756122871, 8.785167769928838, 5.077444009310833, 0.7348414910567824, 2.010329982552813, 8.009785029943638, 3.077833282978799, 4.596971042485385, 8.16411375225135, 5.562235074466959, 8.744478299776523, 9.487104536070332, 1.1297295856319756, 1.6849886022276916, 7.3523944768865075, 8.819691033489644, 0.06271450269784373, 4.05195057838946, 4.724496065558196, 4.777499043577581, 0.6089528950598133, 2.9318553221831154}
#define ADD_Y {8.939979760295667, 9.46527852124558, 7.079237099742524, 4.373761013725459, 0.20169225074546326, 0.9397815688895361, 1.5195375535874267, 5.190426590574847, 5.836610977393284, 5.141679942648915, 4.975245216666565, 6.92932444112015, 6.752741928882401, 5.009008459099654, 4.23924141263875, 1.886881177895442, 9.303019187145798, 0.7121016215967957, 9.708777097726662, 4.207685734119863, 0.3746687430552009, 2.583491003985997, 2.847071448656433, 2.750537473176744, 8.939938676577237, 6.660905332177806, 0.592609687318556, 1.499458197796012, 4.928711135585661, 6.305681055361401, 1.1109129746154445, 7.905063290521056}
#define ADD_Z {16.721317789261537, 11.265363536164893, 9.827951097341252, 10.140371106613777, 8.606134624746035, 2.9615422848815562, 8.081024694529106, 9.512056034973039, 7.10224781201476, 11.327675149523635, 9.986830972789436, 15.714492211048988, 11.830185938193233, 5.743849950156436, 6.249571395191563, 9.896666207839079, 12.380852470124598, 5.309072664082181, 17.872890849978013, 9.769920808586821, 9.119147042831724, 12.070595540056328, 3.9768010342884086, 4.435526075404436, 16.292333153463744, 15.480596365667449, 0.6553241900163997, 5.551408776185472, 9.653207201143857, 11.083180098938982, 1.7198658696752578, 10.83691861270417}

#define MULADD_X {1.343428138574233, 9.913072666182298, 2.4309806042489326, 2.623989939015978, 7.601351459723177, 6.6290365093671735, 0.16554162791492688, 8.366591234405117, 6.178244898597632, 9.760670461598655, 4.188874704248116, 0.48676749334736424, 0.505081880539896, 0.24855690251590956, 6.482669332174421, 6.769796417321827, 5.050795534911295, 7.497692004771293, 6.828015368812904, 1.0046108587327607, 0.379149204252478, 8.668434211942982, 4.381457700322009, 7.795856186615456, 7.553747194365446, 1.6618681603090157, 3.054214367775543, 2.2169756476391242, 0.3796657290793437, 9.817356276887246, 7.31144280894866, 1.5046949108455454}
#define MULADD_Y {1.0698128133817975, 9.369496735788957, 3.7369650042214464, 2.8580738948349227, 5.741290410792877, 6.935847916760776, 9.658357528576332, 6.65155325014029, 9.933867663711437, 1.7243539025179278, 6.091159726047357, 8.19769483708853, 4.338560601050788, 8.962099573748965, 7.620766629882888, 4.880927883946785, 9.590376676426596, 1.0436365162197414, 0.8505768052793738, 9.816942270869083, 6.10063079688656, 6.858039344544654, 6.955348766610307, 7.362414989051161, 8.624058161823374, 6.419460825541423, 0.5498923787580268, 8.014237923285531, 3.85958226247669, 9.941024212888557, 3.6667445755472574, 3.5066436176816773}
#define MULADD_S {3.583132673136925, 2.1007119178351683, 2.6714543086119336, 0.8498683093132187, 4.785974314306869, 3.9917774656392915, 3.302466231033848, 2.75062666603164, 9.526389602539211, 5.3790623944648, 0.1842923066975588, 5.286715349728157, 3.162242737146277, 0.09865849285847772, 9.767124030639224, 1.8421866246432428, 1.8085091310514767, 9.278150276395042, 4.250765902963263, 2.161066124727281, 8.690194989066697, 6.1272005370005145, 7.453721951449128, 2.4551675209579917, 0.0039004219776916926, 2.2403739922773047, 5.712637941749103, 3.8873613287938023, 7.333217485036788, 6.958437077556422, 2.449603525801108, 9.759993399581507}
#define MULADD_Z {5.883494070718652, 30.194006628004157, 10.231218613594299, 5.088119787961328, 42.12116325104703, 33.39748647375281, 10.205053164595748, 29.66492220328158, 68.79023562765285, 54.227609327266634, 6.863137107760297, 10.77109601591674, 5.935752109452258, 8.986621823140757, 70.9378020468516, 17.3521562954948, 18.7247865203877, 70.6083496626134, 29.874871719938394, 11.987972766209635, 9.39551131179007, 59.97127410291533, 39.613516206846285, 26.502547896488853, 8.653520963394204, 10.142667030491472, 17.99751325834784, 16.632423322795457, 6.643753625430573, 78.25448013356203, 21.576880659041052, 18.192456015918086}

// sqrt
float sqrtf(float x);
float ta_sqrt(float x);
float ieee754_sqrtf (float x);
int sqrt_buf(float *input, float *output, int size); 
#define E  2.7182818284590452354

// add
void normal_array_add(float *x, float *y, float *z, int size); 
int neon_array_add(float *x, float *y, float *z, int size);

// muladd
int normal_muladd(float *x, float *s, float *y, float *z, int size);
int neon_muladd(float *x, float *s, float *y, float *z, int size);