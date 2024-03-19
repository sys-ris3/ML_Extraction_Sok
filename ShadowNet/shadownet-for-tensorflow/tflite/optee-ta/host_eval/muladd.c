#include <arm_neon.h>
void normal_muladd(float *x, float *s, float *y, float *z, int size) {
    int i;
    for (i = 0; i < size; i++) {
        z[i] = x[i]*s[i] + y[i];
    }
}

void neon_mul_add_32x4x4(float *x, float *s, float *y, float *z) {
    const float32x4x4_t  xx =
    {
        {
            vld1q_f32(x),
            vld1q_f32(x + 4),
            vld1q_f32(x + 8),
            vld1q_f32(x + 12)
        }
    };
    const float32x4x4_t ss = {
            {
                vld1q_f32(s),
                vld1q_f32(s + 4),
                vld1q_f32(s + 8),
                vld1q_f32(s + 12)
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
    float32x4x4_t zz = {
            {
                vmlaq_f32(yy.val[0], xx.val[0], ss.val[0]),
                vmlaq_f32(yy.val[1], xx.val[1], ss.val[1]),
                vmlaq_f32(yy.val[2], xx.val[2], ss.val[2]),
                vmlaq_f32(yy.val[3], xx.val[3], ss.val[3]) 
            }
        };
    vst1q_f32(z, zz.val[0]);
    vst1q_f32(z + 4, zz.val[1]);
    vst1q_f32(z + 8, zz.val[2]);
    vst1q_f32(z + 12, zz.val[3]);
}

int neon_muladd(float *x, float *s, float *y, float *z, int size) {
    int i;

    if ((size % 16) != 0) {
        return -1;
    }
    
    for (i = 0; i < size/16; i++) {
        neon_mul_add_32x4x4(x+i*16, s+i*16, y+i*16, z+i*16);
    }

    return 0;
}
