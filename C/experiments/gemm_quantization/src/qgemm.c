#include "qgemm.h"

void qgemm(float Sx, int8_t Zx, int8_t* qw, float Sw, int32_t* qb, float Sy, int8_t Zy, int size_out, int size_in, int8_t* res, int8_t* qx) {
    int temp;
    float tempf;
    float c = (Sw * Sx) / Sy;
    for(int i=0; i<size_out; i++) {
        temp = 0;
        for(int j=0; j<size_in; j++) {
            temp += qw[i*size_in+j] * (qx[j]-Zx);        
        }
        temp += qb[i];
        tempf = c * (float)temp;
        res[i] = (int8_t)tempf + Zy;
    }
}