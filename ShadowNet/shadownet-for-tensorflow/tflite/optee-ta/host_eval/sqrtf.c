/* A union which permits us to convert between a float and a 32 bit
   int.  */
typedef unsigned int uint32_t;
typedef int int32_t;
typedef union
{
  float value;
  uint32_t word;
} ieee_float_shape_type;
/* Get a 32 bit int from a float.  */
#ifndef GET_FLOAT_WORD
# define GET_FLOAT_WORD(i,d)                                        \
do {                                                                \
  ieee_float_shape_type gf_u;                                        \
  gf_u.value = (d);                                                \
  (i) = gf_u.word;                                                \
} while (0)
#endif
/* Set a float from a 32 bit int.  */
#ifndef SET_FLOAT_WORD
# define SET_FLOAT_WORD(d,i)                                        \
do {                                                                \
  ieee_float_shape_type sf_u;                                        \
  sf_u.word = (i);                                                \
  (d) = sf_u.value;                                                \
} while (0)
#endif

static const float tiny = 1.0e-30;

float sqrtf(float x)
{
	float z;
	int32_t sign = (int)0x80000000;
	int32_t ix,s,q,m,t,i;
	uint32_t r;

	GET_FLOAT_WORD(ix, x);

	/* take care of Inf and NaN */
	if ((ix&0x7f800000) == 0x7f800000)
		return x*x + x; /* sqrt(NaN)=NaN, sqrt(+inf)=+inf, sqrt(-inf)=sNaN */

	/* take care of zero */
	if (ix <= 0) {
		if ((ix&~sign) == 0)
			return x;  /* sqrt(+-0) = +-0 */
		if (ix < 0)
			return (x-x)/(x-x);  /* sqrt(-ve) = sNaN */
	}
	/* normalize x */
	m = ix>>23;
	if (m == 0) {  /* subnormal x */
		for (i = 0; (ix&0x00800000) == 0; i++)
			ix<<=1;
		m -= i - 1;
	}
	m -= 127;  /* unbias exponent */
	ix = (ix&0x007fffff)|0x00800000;
	if (m&1)  /* odd m, double x to make it even */
		ix += ix;
	m >>= 1;  /* m = [m/2] */

	/* generate sqrt(x) bit by bit */
	ix += ix;
	q = s = 0;       /* q = sqrt(x) */
	r = 0x01000000;  /* r = moving bit from right to left */

	while (r != 0) {
		t = s + r;
		if (t <= ix) {
			s = t+r;
			ix -= t;
			q += r;
		}
		ix += ix;
		r >>= 1;
	}

	/* use floating add to find out rounding direction */
	if (ix != 0) {
		z = 1.0f - tiny; /* raise inexact flag */
		if (z >= 1.0f) {
			z = 1.0f + tiny;
			if (z > 1.0f)
				q += 2;
			else
				q += q & 1;
		}
	}
	ix = (q>>1) + 0x3f000000;
	SET_FLOAT_WORD(z, ix + ((uint32_t)m << 23));
	return z;
}
