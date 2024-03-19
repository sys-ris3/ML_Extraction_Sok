/* Use VFP square root instruction.  */
# include <math.h>
# include <arm_neon.h>

float
ieee754_sqrtf (float s)
{
  float res;
  asm ("fsqrt   %s0, %s1" : "=w" (res) : "w" (s));
  return res;
}
