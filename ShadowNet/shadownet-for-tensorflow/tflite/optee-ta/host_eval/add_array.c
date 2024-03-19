#include <arm_neon.h>

void normal_array_add(float *x, float *y, float *z, int size) {
    int i;
    for (i = 0; i < size; i++) {
        z[i] = x[i] + y[i];
    }
}
	
static inline void neon_add_buf_32x4x4(float *x, float *y, float *z) {
    const float32x4x4_t  xx =
    {
        {
            vld1q_f32(x),
            vld1q_f32(x + 4),
            vld1q_f32(x + 8),
            vld1q_f32(x + 12)
        }
    };
    const float32x4x4_t yy = {
            {
                vld1q_f32(y),
                vld1q_f32(y + 4),
                vld1q_f32(y + 8),
                vld1q_f32(y + 12)
            }
          };
#if 0
    float32x4x4_t zz = {
            {
                vaddq_f32(xx.val[0], yy.val[0]),
                vaddq_f32(xx.val[1], yy.val[1]),
                vaddq_f32(xx.val[2], yy.val[2]),
                vaddq_f32(xx.val[3], yy.val[3])
            }
        };

    vst1q_f32(z, zz.val[0]);
    vst1q_f32(z + 4, zz.val[1]);
    vst1q_f32(z + 8, zz.val[2]);
    vst1q_f32(z + 12, zz.val[3]);
#endif
    vst1q_f32(z, vaddq_f32(xx.val[0], yy.val[0]));
    vst1q_f32(z + 4, vaddq_f32(xx.val[1], yy.val[1]));
    vst1q_f32(z + 8, vaddq_f32(xx.val[2], yy.val[2]));
    vst1q_f32(z + 12, vaddq_f32(xx.val[3], yy.val[3]));
}

int neon_array_add(float *x, float *y, float *z, int size) {
    int i;

    if ((size % 16) != 0) {
        return -1;
    }
    
    for (i = 0; i < size/16; i++) {
        neon_add_buf_32x4x4(x+i*16, y+i*16, z+i*16);
    }

    return 0;
}
