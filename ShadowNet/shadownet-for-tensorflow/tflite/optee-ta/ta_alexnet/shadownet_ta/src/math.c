#include<math_ta.h>
float ta_exp(float x)
{
    if(x<0) return 1/ta_exp(-x);
    int n = (int)x;
    x -= n;
    float e1 = ta_pow(E,n);
    float e2 = ta_eee(x);
    return e1*e2;
}

float ta_eee(float x)
{
    if(x>1e-3)
    {
        float ee = ta_eee(x/2);
        return ee*ee;
    }
    else
        return 1 + x + x*x/2 + ta_pow(x,3)/6 + ta_pow(x,4)/24 + ta_pow(x,5)/120;
}

float ta_pow(float a, int n)
{
    if(n<0) return 1/ta_pow(a,-n);
    float res = 1.0;
    while(n)
    {
        if(n&1) res *= a;
        a *= a;
        n >>= 1;
    }
    return res;
}

float ta_sqrt(float x)
{
    if(x>100) return 10.0*ta_sqrt(x/100);
    float t = x/8 + 0.5 + 2*x/(4+x);
    int c = 10;
    while(c--)
    {
        t = (t+x/t)/2;
    }
    return t;
}
