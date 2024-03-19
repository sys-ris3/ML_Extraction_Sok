#include <arm_neon.h>
void neon_sqrt_buf_32x4x4(float *input_ptr, float *output_ptr); 

inline float32x4_t vinvq_f32(float32x4_t x)
{
    float32x4_t recip = vrecpeq_f32(x);
    recip             = vmulq_f32(vrecpsq_f32(x, recip), recip);
    recip             = vmulq_f32(vrecpsq_f32(x, recip), recip);
    return recip;
}

inline float32x4_t vinvsqrtq_f32(float32x4_t x)
{
    float32x4_t sqrt_reciprocal = vrsqrteq_f32(x);
    sqrt_reciprocal             = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal);
    sqrt_reciprocal             = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal);

    return sqrt_reciprocal;
}

int sqrt_buf(float *input, float *output, int size) {
    int i;

    if ((size % 16) != 0) {
        return -1;
    }
    
    for (i = 0; i < size/16; i++) {
        neon_sqrt_buf_32x4x4(input+i*16, output+i*16);
    }

    return 0;
}

void neon_sqrt_buf_32x4x4(float *input_ptr, float *output_ptr) {
    const float32x4x4_t in =
    {
        {
            vld1q_f32(input_ptr),
            vld1q_f32(input_ptr + 4),
            vld1q_f32(input_ptr + 8),
            vld1q_f32(input_ptr + 12)
        }
    };
    float32x4x4_t tmp = {
            {
                vinvq_f32(vinvsqrtq_f32(in.val[0])),
                vinvq_f32(vinvsqrtq_f32(in.val[1])),
                vinvq_f32(vinvsqrtq_f32(in.val[2])),
                vinvq_f32(vinvsqrtq_f32(in.val[3])),
            }
          };

    vst1q_f32(output_ptr, tmp.val[0]);
    vst1q_f32(output_ptr + 4, tmp.val[1]);
    vst1q_f32(output_ptr + 8, tmp.val[2]);
    vst1q_f32(output_ptr + 12, tmp.val[3]);
}
