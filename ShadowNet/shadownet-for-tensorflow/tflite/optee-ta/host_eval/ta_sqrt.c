float ta_sqrt(float x);
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

