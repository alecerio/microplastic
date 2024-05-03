#include "quant_helper.h"

#define INPUT_SIZE (4)

#define EXP_LUT_SIZE (256)
static float EXP_LUT_X[EXP_LUT_SIZE] = {
    -10.0f, -9.921875, -9.84375, -9.765625, -9.6875, -9.609375,
  -9.53125, -9.453125, -9.375, -9.296875, -9.21875, -9.140625,
  -9.0625, -8.984375, -8.90625, -8.828125, -8.75, -8.671875,
  -8.59375, -8.515625, -8.4375, -8.359375, -8.28125, -8.203125,
  -8.125, -8.046875, -7.96875, -7.890625, -7.8125, -7.734375,
  -7.65625, -7.578125, -7.5, -7.421875, -7.34375, -7.265625,
  -7.1875, -7.109375, -7.03125, -6.953125, -6.875, -6.796875,
  -6.71875, -6.640625, -6.5625, -6.484375, -6.40625, -6.328125,
  -6.25, -6.171875, -6.09375, -6.015625, -5.9375, -5.859375,
  -5.78125, -5.703125, -5.625, -5.546875, -5.46875, -5.390625,
  -5.3125, -5.234375, -5.15625, -5.078125, -5., -4.921875,
  -4.84375, -4.765625, -4.6875, -4.609375, -4.53125, -4.453125,
  -4.375, -4.296875, -4.21875, -4.140625, -4.0625, -3.984375,
  -3.90625,   -3.828125,  -3.75,      -3.671875,  -3.59375,   -3.515625,
  -3.4375,    -3.359375,  -3.28125,   -3.203125,  -3.125,     -3.046875,
  -2.96875,   -2.890625,  -2.8125,    -2.734375,  -2.65625,   -2.578125,
  -2.5,       -2.421875,  -2.34375,   -2.265625,  -2.1875,    -2.109375,
  -2.03125,   -1.953125,  -1.875,     -1.796875,  -1.71875,   -1.640625,
  -1.5625,    -1.484375,  -1.40625,   -1.328125,  -1.25,      -1.171875,
  -1.09375,   -1.015625,  -0.9375,    -0.859375,  -0.78125,   -0.703125,
  -0.625,     -0.546875,  -0.46875,   -0.390625,  -0.3125,    -0.234375,
  -0.15625,   -0.078125,   0.,         0.078125,   0.15625,    0.234375,
   0.3125,     0.390625,   0.46875,    0.546875,   0.625,      0.703125,
   0.78125,    0.859375,   0.9375,     1.015625,   1.09375,    1.171875,
   1.25,       1.328125,   1.40625,    1.484375,   1.5625,     1.640625,
   1.71875,    1.796875,   1.875,      1.953125,   2.03125,    2.109375,
   2.1875,     2.265625,   2.34375,    2.421875,   2.5,        2.578125,
   2.65625,    2.734375,   2.8125,     2.890625,   2.96875,    3.046875,
   3.125,      3.203125,   3.28125,    3.359375,   3.4375,     3.515625,
   3.59375,    3.671875,   3.75,       3.828125,   3.90625,    3.984375,
   4.0625,     4.140625,   4.21875,    4.296875,   4.375,      4.453125,
   4.53125,    4.609375,   4.6875,     4.765625,   4.84375,    4.921875,
   5.,         5.078125,   5.15625,    5.234375,   5.3125,     5.390625,
   5.46875,    5.546875,   5.625,      5.703125,   5.78125,    5.859375,
   5.9375,     6.015625,   6.09375,    6.171875,   6.25,       6.328125,
   6.40625,    6.484375,   6.5625,     6.640625,   6.71875,    6.796875,
   6.875,      6.953125,   7.03125,    7.109375,   7.1875,     7.265625,
   7.34375,    7.421875,   7.5,        7.578125,   7.65625,    7.734375,
   7.8125,     7.890625,   7.96875,    8.046875,   8.125,      8.203125,
   8.28125,    8.359375,   8.4375,     8.515625,   8.59375,    8.671875,
   8.75,       8.828125,   8.90625,    8.984375,   9.0625,     9.140625,
   9.21875,    9.296875,   9.375,      9.453125,   9.53125,    9.609375,
   9.6875,     9.765625,   9.84375,    9.921875,
};
static float EXP_LUT_Y[EXP_LUT_SIZE] = {
 4.53999310e-05f, 4.90890270e-05f, 5.30778962e-05f, 5.73908874e-05f,
 6.20543433e-05f, 6.70967493e-05f, 7.25488790e-05f, 7.84440417e-05f,
 8.48182317e-05f, 9.17103825e-05f, 9.91625639e-05f, 1.07220294e-04f,
 1.15932780e-04f, 1.25353225e-04f, 1.35539158e-04f, 1.46552775e-04f,
 1.58461320e-04f, 1.71337539e-04f, 1.85260054e-04f, 2.00313880e-04f,
 2.16590954e-04f, 2.34190651e-04f, 2.53220467e-04f, 2.73796613e-04f,
 2.96044716e-04f, 3.20100662e-04f, 3.46111367e-04f, 3.74235591e-04f,
 4.04645165e-04f, 4.37525741e-04f, 4.73078137e-04f, 5.11519436e-04f,
 5.53084363e-04f, 5.98026789e-04f, 6.46621163e-04f, 6.99164171e-04f,
 7.55976711e-04f, 8.17405700e-04f, 8.83826287e-04f, 9.55644122e-04f,
 1.03329762e-03f, 1.11726113e-03f, 1.20804738e-03f, 1.30621064e-03f,
 1.41235045e-03f, 1.52711489e-03f, 1.65120489e-03f, 1.78537820e-03f,
 1.93045416e-03f, 2.08731857e-03f, 2.25692964e-03f, 2.44032266e-03f,
 2.63861800e-03f, 2.85302638e-03f, 3.08485702e-03f, 3.33552575e-03f,
 3.60656320e-03f, 3.89962457e-03f, 4.21649963e-03f, 4.55912296e-03f,
 4.92958724e-03f, 5.33015467e-03f, 5.76327136e-03f, 6.23158226e-03f,
 6.73794700e-03f, 7.28545757e-03f, 7.87745789e-03f, 8.51756334e-03f,
 9.20968130e-03f, 9.95803997e-03f, 1.07672084e-02f, 1.16421282e-02f,
 1.25881424e-02f, 1.36110270e-02f, 1.47170294e-02f, 1.59129035e-02f,
 1.72059499e-02f, 1.86040681e-02f, 2.01157946e-02f, 2.17503589e-02f,
 2.35177465e-02f, 2.54287459e-02f, 2.74950303e-02f, 2.97292173e-02f,
 3.21449488e-02f, 3.47569771e-02f, 3.75812501e-02f, 4.06350195e-02f,
 4.39369343e-02f, 4.75071520e-02f, 5.13674803e-02f, 5.55414893e-02f,
 6.00546673e-02f, 6.49345815e-02f, 7.02110231e-02f, 7.59162158e-02f,
 8.20849985e-02f, 8.87550488e-02f, 9.59670842e-02f, 1.03765160e-01f,
 1.12196892e-01f, 1.21313766e-01f, 1.31171450e-01f, 1.41830161e-01f,
 1.53354973e-01f, 1.65816262e-01f, 1.79290116e-01f, 1.93858847e-01f,
 2.09611386e-01f, 2.26643950e-01f, 2.45060533e-01f, 2.64973611e-01f,
 2.86504805e-01f, 3.09785545e-01f, 3.34958047e-01f, 3.62176001e-01f,
 3.91605616e-01f, 4.23426628e-01f, 4.57833350e-01f, 4.95035887e-01f,
 5.35261452e-01f, 5.78755617e-01f, 6.25783980e-01f, 6.76633835e-01f,
 7.31615603e-01f, 7.91065097e-01f, 8.55345309e-01f, 9.24848795e-01f,
 1.00000000e+00f, 1.08125782e+00f, 1.16911840e+00f, 1.26411843e+00f,
 1.36683798e+00f, 1.47790420e+00f, 1.59799540e+00f, 1.72784507e+00f,
 1.86824596e+00f, 2.02005553e+00f, 2.18420076e+00f, 2.36168408e+00f,
 2.55358934e+00f, 2.76108861e+00f, 2.98544860e+00f, 3.22803950e+00f,
 3.49034286e+00f, 3.77396059e+00f, 4.08062410e+00f, 4.41220713e+00f,
 4.77073336e+00f, 5.15839243e+00f, 5.57755232e+00f, 6.03077173e+00f,
 6.52081919e+00f, 7.05068636e+00f, 7.62361002e+00f, 8.24308777e+00f,
 8.91290283e+00f, 9.63714600e+00f, 1.04202394e+01f, 1.12669649e+01f,
 1.21824942e+01f, 1.31724167e+01f, 1.42427788e+01f, 1.54001150e+01f,
 1.66514950e+01f, 1.80045586e+01f, 1.94675694e+01f, 2.10494614e+01f,
 2.27598953e+01f, 2.46093140e+01f, 2.66090126e+01f, 2.87712040e+01f,
 3.11090889e+01f, 3.36369438e+01f, 3.63702087e+01f, 3.93255730e+01f,
 4.25210838e+01f, 4.59762535e+01f, 4.97121811e+01f, 5.37516823e+01f,
 5.81194267e+01f, 6.28420868e+01f, 6.79484940e+01f, 7.34698410e+01f,
 7.94398422e+01f, 8.58949432e+01f, 9.28745804e+01f, 1.00421364e+02f,
 1.08581390e+02f, 1.17404472e+02f, 1.26944504e+02f, 1.37259735e+02f,
 1.48413162e+02f, 1.60472885e+02f, 1.73512558e+02f, 1.87611816e+02f,
 2.02856735e+02f, 2.19340424e+02f, 2.37163559e+02f, 2.56434937e+02f,
 2.77272278e+02f, 2.99802826e+02f, 3.24164154e+02f, 3.50505005e+02f,
 3.78986267e+02f, 4.09781860e+02f, 4.43079834e+02f, 4.79083557e+02f,
 5.18012817e+02f, 5.60105408e+02f, 6.05618347e+02f, 6.54829590e+02f,
 7.08039612e+02f, 7.65573303e+02f, 8.27782166e+02f, 8.95045898e+02f,
 9.67775391e+02f, 1.04641467e+03f, 1.13144409e+03f, 1.22338269e+03f,
 1.32279211e+03f, 1.43027930e+03f, 1.54650061e+03f, 1.67216589e+03f,
 1.80804236e+03f, 1.95495996e+03f, 2.11381567e+03f, 2.28557983e+03f,
 2.47130103e+03f, 2.67211353e+03f, 2.88924365e+03f, 3.12401709e+03f,
 3.37786792e+03f, 3.65234619e+03f, 3.94912769e+03f, 4.27002539e+03f,
 4.61699805e+03f, 4.99216504e+03f, 5.39781738e+03f, 5.83643262e+03f,
 6.31068799e+03f, 6.82348096e+03f, 7.37794189e+03f, 7.97745703e+03f,
 8.62568750e+03f, 9.32659277e+03f, 1.00844512e+04f, 1.09038916e+04f,
 1.17899180e+04f, 1.27479404e+04f, 1.37838105e+04f, 1.49038525e+04f,
 1.61149062e+04f, 1.74243691e+04f, 1.88402344e+04f, 2.03711504e+04f,
};

float access_softmax_lut(float x);

int main() {
    
    float x[INPUT_SIZE] = {
        0.0f, -0.01f, 0.02f, -0.03f
    };

    int8_t qx[INPUT_SIZE];
    quantize(x, qx, INPUT_SIZE, 0.003917609341442585, 0);

    float dq[INPUT_SIZE];
    dequantize(qx, dq, INPUT_SIZE, 0.003917609341442585, 0);


    float exps[INPUT_SIZE];
    float sum_exps = 0.0f;
    for(int i=0; i<INPUT_SIZE; i++) {
        exps[i] = access_softmax_lut(dq[i]);
        sum_exps += exps[i];
    }
    float sm[INPUT_SIZE];
    for(int i=0; i<INPUT_SIZE; i++) {
        sm[i] = exps[i] / sum_exps;
    }

    quantize(sm, qx, INPUT_SIZE, 0.003917609341442585, 0);

    dequantize(qx, dq, INPUT_SIZE, 0.003917609341442585, 0);
    PRINT_MAT(dq, 1, INPUT_SIZE, %f, "OUTPUT");

    return 0;
}

float access_softmax_lut(float x) {
    if(x < EXP_LUT_X[0])
        return EXP_LUT_Y[0];
    else if(x > EXP_LUT_X[EXP_LUT_SIZE-1])
        return EXP_LUT_Y[EXP_LUT_SIZE-1];
    else {
        for(int i=0; i<EXP_LUT_SIZE; i++) {
            if(x >= EXP_LUT_X[i] && x < EXP_LUT_X[i+1]) {
                float x0 = EXP_LUT_X[i];
                float x1 = EXP_LUT_X[i+1];
                float y0 = EXP_LUT_Y[i];
                float y1 = EXP_LUT_Y[i+1];
                return y0 + (y1 - y0) * ((x - x0) / (x1 - x0));
            }
        }
    }
}