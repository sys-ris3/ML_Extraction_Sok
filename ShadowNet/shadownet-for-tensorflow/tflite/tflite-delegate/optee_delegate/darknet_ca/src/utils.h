#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>
#include <time.h>
#include <android/log.h>
#include "darknet.h"
#include "list.h"

#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "com.tflitedemo ZC native", __VA_ARGS__)
//#define DEBUG_WEIGHTS

#ifdef DEBUG_WEIGHTS
/* dump weigths first 8 bytes */
#define DUMPW4F(TAG, size, pw) fprintf(stderr, TAG " bufsize:%d [0-7]: %6f %6f %6f %6f\n",size, \
        pw[0], \
        pw[1], \
        pw[2], \
        pw[3])

#define DUMPW4I(TAG, size, pw) fprintf(stderr, TAG " bufsize:%d [0-7]: %6d %6d %6d %6d\n",size, \
        pw[0], \
        pw[1], \
        pw[2], \
        pw[3]) 

/* dump weigths first 8 bytes */
#define DUMPW8(TAG, size, pw) fprintf(stderr, TAG " bufsize:%d [0-7]: %3u %3u %3u %3u %3u %3u %3u %3u\n",size, \
        ((unsigned char *)pw)[0], \
        ((unsigned char *)pw)[1], \
        ((unsigned char *)pw)[2], \
        ((unsigned char *)pw)[3], \
        ((unsigned char *)pw)[4], \
        ((unsigned char *)pw)[5], \
        ((unsigned char *)pw)[6], \
        ((unsigned char *)pw)[7])

#define DUMPW4(TAG, size, pw) fprintf(stderr, TAG " bufsize:%d [0-7]: %3u %3u %3u %3u\n",size, \
        ((unsigned char *)pw)[0], \
        ((unsigned char *)pw)[1], \
        ((unsigned char *)pw)[2], \
        ((unsigned char *)pw)[3]) 
#else

#define DUMPW4F(TAG, size, pw) 
#define DUMPW4I(TAG, size, pw) 
#define DUMPW8(TAG, size, pw)
#define DUMPW4(TAG, size, pw)

#endif

#define TIME(a) \
    do { \
    double start = what_time_is_it_now(); \
    a; \
    printf("%s took: %f seconds\n", #a, what_time_is_it_now() - start); \
    } while (0)

#define TWO_PI 6.2831853071795864769252866f

double what_time_is_it_now();
void shuffle(void *arr, size_t n, size_t size);
void sorta_shuffle(void *arr, size_t n, size_t size, size_t sections);
void free_ptrs(void **ptrs, int n);
int alphanum_to_int(char c);
char int_to_alphanum(int i);
int read_int(int fd);
void write_int(int fd, int n);
void read_all(int fd, char *buffer, size_t bytes);
void write_all(int fd, char *buffer, size_t bytes);
int read_all_fail(int fd, char *buffer, size_t bytes);
int write_all_fail(int fd, char *buffer, size_t bytes);
void find_replace(char *str, char *orig, char *rep, char *output);
void malloc_error();
void file_error(char *s);
void strip(char *s);
void strip_char(char *s, char bad);
list *split_str(char *s, char delim);
char *fgetl(FILE *fp);
list *parse_csv_line(char *line);
char *copy_string(char *s);
int count_fields(char *line);
float *parse_fields(char *line, int n);
void translate_array(float *a, int n, float s);
float constrain(float min, float max, float a);
int constrain_int(int a, int min, int max);
float rand_scale(float s);
int rand_int(int min, int max);
void mean_arrays(float **a, int n, int els, float *avg);
float dist_array(float *a, float *b, int n, int sub);
float **one_hot_encode(float *a, int n, int k);
float sec(clock_t clocks);
void print_statistics(float *a, int n);
int int_index(int *a, int val, int n);

#endif

