// *****************************************************************************
// 	THIS CODE WAS AUTOMATICALLY GENERATED ON 2024-06-15 12:07:25
// *****************************************************************************

#include "qgru_model.h"

static float32_t tensor_ortshared_1_0_1_0_token_66[1] = {1.0, };

static int8_t tensor_W_hzGemm_output_0_zero_point[1] = {37, };

static float32_t tensor_W_hzGemm_output_0_scale[1] = {0.006902203, };

static int8_t tensor_onnxGemm_1_zero_point[1] = {-128, };

static float32_t tensor_onnxGemm_1_scale[1] = {0.0039179265, };

static float32_t tensor_W_hzweight_scale[1] = {0.0027732097, };

static int8_t tensor_W_hzweight_zero_point[1] = {0, };

static int8_t tensor_W_hzweight_quantized[64] = {2, 36, -95, 11, 68, 70, 85, -30, -15, 87, 118, -107, -64, -105, 66, -56, -111, 63, -78, -66, 61, 122, 67, -29, -75, -14, -28, -74, -27, -127, -2, -11, 55, 69, -21, 12, 69, -16, 80, -101, 109, -67, -58, 45, -41, 3, -51, 57, 118, 113, 9, 82, 41, 29, 10, -57, -96, -101, -32, 117, -126, 30, -106, -63, };

static int32_t tensor_ortshared_1_1_8_0_token_65_quantized[8] = {7796, -24574, -16688, 3985, -8342, -21325, -18026, -23674, };

static int8_t tensor_W_izGemm_output_0_zero_point[1] = {6, };

static float32_t tensor_W_izGemm_output_0_scale[1] = {0.008279371, };

static int8_t tensor_onnxGemm_0_zero_point[1] = {-128, };

static float32_t tensor_onnxGemm_0_scale[1] = {0.0039189, };

static float32_t tensor_W_izweight_scale[1] = {0.0012289806, };

static int8_t tensor_W_izweight_zero_point[1] = {0, };

static int8_t tensor_W_izweight_quantized[328] = {-103, 84, 33, -86, 16, -84, 75, 96, 108, 14, -95, 54, 54, 42, 89, 34, -46, -50, -50, 42, 123, -77, 77, -76, -116, -14, -15, 86, 45, -49, 76, 30, 101, 127, 20, -17, 67, 54, 56, -124, -67, -52, 81, 21, -21, 100, -112, 20, 113, 51, 56, -70, -47, -22, -64, 54, -125, 40, -8, 118, -25, 112, -117, -119, 96, 104, 12, -114, 27, -6, 69, -115, 115, 102, -96, -27, -126, -16, 127, -62, 2, 60, 79, 3, -40, -6, 30, -59, -10, 65, -22, -56, -62, 70, -74, 37, -28, 15, 13, 110, 11, 28, 84, 24, 123, -89, 88, -32, -84, 124, 65, -81, 110, 123, 17, 73, -37, 93, 15, -7, 23, 121, -53, -47, 74, -108, 7, -73, -121, 101, -88, 60, -67, -124, -45, -120, -40, 12, -17, -86, 39, -26, -109, -61, 109, -101, 66, 24, -89, 87, 115, 109, -99, 70, -1, -86, 102, -74, -93, -41, -95, -122, -65, 118, -99, 14, 45, -14, -124, -109, 54, 14, 117, -120, -90, 8, -102, -20, -59, -65, 67, -25, 4, -124, -115, 118, 38, -78, 101, 99, 62, -24, 34, -16, 21, 40, -69, 118, -10, 27, -14, -25, -87, 62, -20, -57, -7, -31, 44, 87, 90, -32, 110, -87, -88, 110, -53, 24, -31, 60, 74, 72, 94, -109, 38, -44, -12, -17, -25, -77, 53, 49, -114, 118, 92, 2, -26, -77, -88, -54, -111, 53, 50, 111, 18, 80, -42, 90, -110, -71, 30, 64, -99, -50, -20, 7, 3, -67, -1, -31, 36, -99, 68, 29, -55, 99, 105, -29, 63, 20, -73, 124, -1, -69, 27, 96, -25, 114, -12, -81, -45, 73, 94, 30, -28, 38, -68, -15, 20, -37, -14, 33, 35, 109, -44, 126, -100, 17, -81, -83, 2, -58, 93, -61, -15, 64, 103, 73, -2, -88, 44, 33, -38, -65, 21, 124, 75, -86, -3, 88, -42, -88, -61, 117, 111, -24, 56, 64, };

static int32_t tensor_ortshared_1_1_8_1_token_67_quantized[8] = {-13873, -28160, 32048, -5317, -6082, 30882, 32360, 21190, };

static int8_t tensor_Add_1_output_0_zero_point[1] = {20, };

static float32_t tensor_Add_1_output_0_scale[1] = {0.010285696, };

static int8_t tensor_Sigmoid_1_output_0_zero_point[1] = {-128, };

static float32_t tensor_Sigmoid_1_output_0_scale[1] = {0.0029432562, };

static int8_t tensor_Mul_2_output_0_zero_point[1] = {-128, };

static float32_t tensor_Mul_2_output_0_scale[1] = {0.0027144453, };

static int8_t tensor_W_hnGemm_output_0_zero_point[1] = {-25, };

static float32_t tensor_W_hnGemm_output_0_scale[1] = {0.005947366, };

static float32_t tensor_W_hnweight_scale[1] = {0.002773485, };

static int8_t tensor_W_hnweight_zero_point[1] = {0, };

static int8_t tensor_W_hnweight_quantized[64] = {-5, 68, -28, 100, -126, -15, 85, -11, -76, -13, 112, -58, 1, 54, 67, 94, -16, -9, -55, 91, 37, 79, 101, -55, 85, -37, 55, 13, 127, -109, 16, -86, -19, 65, 42, 95, -18, -34, -104, -12, 92, -3, 58, 62, -72, -127, 39, -119, -94, -104, 69, 52, -4, 103, 20, -11, -95, 110, 108, 63, -81, -71, -127, 65, };

static int32_t tensor_ortshared_1_1_8_3_token_69_quantized[8] = {20384, -6595, 31377, 16247, -27144, 3344, 5379, 17724, };

static int8_t tensor_W_hrGemm_output_0_zero_point[1] = {36, };

static float32_t tensor_W_hrGemm_output_0_scale[1] = {0.00958655, };

static float32_t tensor_W_hrweight_scale[1] = {0.0027780046, };

static int8_t tensor_W_hrweight_zero_point[1] = {0, };

static int8_t tensor_W_hrweight_quantized[64] = {11, -4, -83, -58, -78, -47, 97, 74, 57, -44, -48, 35, 124, 58, -89, 66, 57, -117, 2, 27, -77, 96, 74, 75, -66, 68, -59, 52, 50, -99, 20, -17, -59, -127, -113, -79, -57, -28, -53, -105, -122, 3, -88, 39, -4, -121, -13, -88, -31, -12, -73, 70, 104, -122, 61, 55, 49, 10, 54, -7, -122, 54, 110, -96, };

static int32_t tensor_ortshared_1_1_8_4_token_70_quantized[8] = {4944, 20533, -25040, 15495, -20453, -9808, 13977, 16554, };

static int8_t tensor_W_irGemm_output_0_zero_point[1] = {-25, };

static float32_t tensor_W_irGemm_output_0_scale[1] = {0.008226714, };

static float32_t tensor_W_irweight_scale[1] = {0.0012161068, };

static int8_t tensor_W_irweight_zero_point[1] = {0, };

static int8_t tensor_W_irweight_quantized[328] = {9, 90, 113, 4, -45, -28, -67, -25, -121, -35, -11, 32, 85, -6, -122, 45, -57, 38, 111, 12, 3, -13, 124, 33, 89, -60, -22, -82, 48, -10, 38, 60, 17, 102, 108, 63, 60, 59, -14, -32, -33, -73, -52, 59, 9, 58, -60, 107, -115, 59, -8, 83, -48, 97, -8, 45, 0, -121, -98, -10, 16, 91, -75, 29, 114, 103, 119, -17, 5, 14, -70, 59, 102, 114, 125, 32, 107, 124, 100, 45, 117, -4, -25, -106, -23, -14, 17, 33, -113, -97, 29, -95, 74, 63, -84, -76, 45, -5, 30, 3, 37, -56, -39, 87, -15, -48, 16, -98, -127, 111, 43, 114, 67, -114, -25, 123, -104, 54, -60, -12, 39, -21, 36, -119, -101, 102, 15, 103, -79, 47, -33, 39, -16, -5, -107, 84, 80, -8, -66, -58, -116, 97, 95, 101, -49, -42, -29, -110, -121, -29, -63, 27, 100, -29, -57, -125, 69, -96, 57, -60, -15, 104, 84, -37, -73, -107, -113, -126, 64, -5, -88, -61, -31, -82, -2, 115, 99, 10, -34, 5, -96, 116, 84, -98, -102, -88, -125, 10, 23, -107, -121, -42, 106, 102, -58, 6, -31, 104, -28, 30, 66, 41, -17, 35, -118, 36, 102, 38, 9, -82, -46, -76, 81, 119, 66, -18, 67, 102, -46, 85, 45, 19, 65, -105, 68, -18, -77, 97, -114, -125, -96, -117, 21, -64, -31, 90, 22, 19, -39, 70, -37, 32, -34, -43, 123, 55, 74, 69, 41, -49, 74, 21, 3, -7, 95, -126, -73, 26, 67, 0, 11, 100, 121, -19, 32, 125, 27, 50, 35, 103, -113, -38, 122, 62, -21, 99, 97, -106, 66, -110, -58, 87, 120, -33, -58, 110, 20, -84, 54, -21, -60, 54, -104, -74, 51, -63, 64, 49, -90, -94, 77, -118, -46, -72, -10, 88, 68, -107, 81, -64, 116, 39, -100, 73, 94, 60, -54, -74, 76, 79, 31, 15, 108, -7, -41, -125, -100, 47, };

static int32_t tensor_ortshared_1_1_8_2_token_68_quantized[8] = {28798, 15250, -31090, -6543, 29764, -31015, -31265, -28202, };

static int8_t tensor_Add_output_0_zero_point[1] = {2, };

static float32_t tensor_Add_output_0_scale[1] = {0.014976039, };

static int8_t tensor_Sigmoid_output_0_zero_point[1] = {-128, };

static float32_t tensor_Sigmoid_output_0_scale[1] = {0.0033969819, };

static int8_t tensor_Mul_output_0_zero_point[1] = {-48, };

static float32_t tensor_Mul_output_0_scale[1] = {0.0024732095, };

static int8_t tensor_W_inGemm_output_0_zero_point[1] = {-10, };

static float32_t tensor_W_inGemm_output_0_scale[1] = {0.0066473405, };

static float32_t tensor_W_inweight_scale[1] = {0.0012282039, };

static int8_t tensor_W_inweight_zero_point[1] = {0, };

static int8_t tensor_W_inweight_quantized[328] = {-101, 13, 6, 43, 127, -24, 87, 52, -76, -123, -49, 102, 45, -14, 114, 4, -93, 59, 123, 1, -39, 93, 35, -28, -123, 94, -11, -89, 31, 59, -55, 58, -1, 119, -67, 98, -36, 91, 49, 98, -69, 38, 88, 42, -11, 81, -91, 2, -61, -86, -70, 119, -29, -114, 8, -50, 28, -104, -84, 23, 85, -11, 100, -122, -121, -108, 37, -82, 79, -69, 51, 79, 86, -2, 24, -3, 51, 21, -112, 126, 83, -31, -70, 60, 84, -122, -104, -24, -74, 113, -114, 112, 59, -120, 13, -117, -103, 116, 35, 28, 36, 95, 94, 55, -90, -20, -83, -33, -103, -107, -30, 51, 15, -66, 119, 85, -36, -94, 80, 52, 109, 55, -17, 87, -103, -2, 97, -61, 65, 73, 109, 79, -32, 63, 113, -100, 96, -90, 57, -124, 55, -49, -68, -102, -8, 6, 12, 50, -14, 65, 52, -120, 21, -109, -8, -49, -77, -18, -96, -45, -13, 49, 43, -56, 72, 101, -21, -91, 16, 102, -65, -23, -82, 116, 83, -40, -120, -102, 75, -50, 78, 45, -11, -64, 122, 56, -18, -21, 53, -47, -113, -72, -72, -72, 63, -50, -115, 69, -29, 12, 11, -101, 6, -120, 0, 55, -81, 114, 84, -81, -126, -10, -52, -6, 78, -48, -103, 19, 107, 120, -85, 8, -2, -111, 105, 126, 66, -25, -50, -18, 90, 60, 100, 11, -57, -55, 43, 106, 65, 101, -99, 59, 85, -124, 6, -2, -8, -52, 73, 122, 68, -80, 10, 105, 45, -15, 1, 117, 99, -37, 33, 89, -65, -47, 111, 74, -72, -72, -66, 89, -97, 50, 108, -87, 42, 83, -98, -11, -13, 104, -127, 7, -120, -80, -107, 21, 39, 24, -41, 76, -125, 36, -80, -75, -28, -113, 61, -37, -92, -42, 85, -126, -46, 22, 104, 21, 30, 100, 116, 48, -21, 123, 125, 1, -78, 23, -65, 43, -81, 112, -61, 114, 2, -83, 13, 104, 14, 87, };

static int32_t tensor_ortshared_1_1_8_5_token_71_quantized[8] = {19673, -18085, 4911, 31756, 39, 29280, -26606, 2910, };

static int8_t tensor_Add_2_output_0_zero_point[1] = {-18, };

static float32_t tensor_Add_2_output_0_scale[1] = {0.0076211523, };

static int8_t tensor_Mul_1_output_0_zero_point[1] = {-8, };

static float32_t tensor_Mul_1_output_0_scale[1] = {0.0028152678, };

static int8_t tensor_Sub_output_0_zero_point[1] = {-128, };

static float32_t tensor_Sub_output_0_scale[1] = {0.0032186117, };

static int8_t tensor_Tanh_output_0_zero_point[1] = {-11, };

static float32_t tensor_Tanh_output_0_scale[1] = {0.005830716, };

static int8_t tensor_Add_3_output_0_zero_point[1] = {-62, };

static float32_t tensor_Add_3_output_0_scale[1] = {0.0046928357, };

static int8_t tensor_output_zero_point[1] = {127, };

static float32_t tensor_output_scale[1] = {0.002316353, };

static float32_t tensor_fcweight_scale[1] = {0.0025500322, };

static int8_t tensor_fcweight_zero_point[1] = {0, };

static int8_t tensor_fcweight_quantized[16] = {-41, -44, -127, -14, -25, -84, 115, -49, 15, 105, -68, -34, -70, -118, 11, 9, };

static int32_t tensor_ortshared_1_1_2_0_token_64_quantized[2] = {-8457, -21953, };


// -------------------------------------------------------
//                         QLIN onnxGemm_0_QuantizeLinear
// -------------------------------------------------------

#include "ncquant.h"

#define QLIN_onnxGemm_0_QuantizeLinear_OUTPUT_SIZE (41)

#define CONNECTED_OUTPUT
#ifdef CONNECTED_OUTPUT
static int8_t tensor_onnxGemm_0_quantized[QLIN_onnxGemm_0_QuantizeLinear_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

// -------------------------------------------------------
//                         QLIN onnxGemm_1_QuantizeLinear
// -------------------------------------------------------

#include "ncquant.h"

#define QLIN_onnxGemm_1_QuantizeLinear_OUTPUT_SIZE (8)

#define CONNECTED_OUTPUT
#ifdef CONNECTED_OUTPUT
static int8_t tensor_onnxGemm_1_quantized[QLIN_onnxGemm_1_QuantizeLinear_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

// -------------------------------------------------------
//                         QGEMM W_izGemm_quant
// -------------------------------------------------------

#define QGEMM_W_izGemm_quant_OUTPUT_SIZE (8)
#define QGEMM_W_izGemm_quant_INPUT_SIZE (41)

#define CONNECTED_OUTPUT
#ifdef CONNECTED_OUTPUT
static int8_t tensor_W_izGemm_output_0_quantized[QGEMM_W_izGemm_quant_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

// -------------------------------------------------------
//                         QGEMM W_irGemm_quant
// -------------------------------------------------------

#define QGEMM_W_irGemm_quant_OUTPUT_SIZE (8)
#define QGEMM_W_irGemm_quant_INPUT_SIZE (41)

#define CONNECTED_OUTPUT
#ifdef CONNECTED_OUTPUT
static int8_t tensor_W_irGemm_output_0_quantized[QGEMM_W_irGemm_quant_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

// -------------------------------------------------------
//                         QGEMM W_inGemm_quant
// -------------------------------------------------------

#define QGEMM_W_inGemm_quant_OUTPUT_SIZE (8)
#define QGEMM_W_inGemm_quant_INPUT_SIZE (41)

#define CONNECTED_OUTPUT
#ifdef CONNECTED_OUTPUT
static int8_t tensor_W_inGemm_output_0_quantized[QGEMM_W_inGemm_quant_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

// -------------------------------------------------------
//                         QGEMM W_hzGemm_quant
// -------------------------------------------------------

#define QGEMM_W_hzGemm_quant_OUTPUT_SIZE (8)
#define QGEMM_W_hzGemm_quant_INPUT_SIZE (8)

#define CONNECTED_OUTPUT
#ifdef CONNECTED_OUTPUT
static int8_t tensor_W_hzGemm_output_0_quantized[QGEMM_W_hzGemm_quant_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

// -------------------------------------------------------
//                         QGEMM W_hnGemm_quant
// -------------------------------------------------------

#define QGEMM_W_hnGemm_quant_OUTPUT_SIZE (8)
#define QGEMM_W_hnGemm_quant_INPUT_SIZE (8)

#define CONNECTED_OUTPUT
#ifdef CONNECTED_OUTPUT
static int8_t tensor_W_hnGemm_output_0_quantized[QGEMM_W_hnGemm_quant_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

// -------------------------------------------------------
//                         QGEMM W_hrGemm_quant
// -------------------------------------------------------

#define QGEMM_W_hrGemm_quant_OUTPUT_SIZE (8)
#define QGEMM_W_hrGemm_quant_INPUT_SIZE (8)

#define CONNECTED_OUTPUT
#ifdef CONNECTED_OUTPUT
static int8_t tensor_W_hrGemm_output_0_quantized[QGEMM_W_hrGemm_quant_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

// -------------------------------------------------------
//                   QLINEARADD Add_1_quant
// -------------------------------------------------------

#define QLINEARADD_Add_1_quant_OUTPUT_SIZE (8)

#define CONNECTED_OUTPUT
#ifdef CONNECTED_OUTPUT
static int8_t tensor_Add_1_output_0_quantized[QLINEARADD_Add_1_quant_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

// -------------------------------------------------------
//                   QLINEARADD Add_quant
// -------------------------------------------------------

#define QLINEARADD_Add_quant_OUTPUT_SIZE (8)

#define CONNECTED_OUTPUT
#ifdef CONNECTED_OUTPUT
static int8_t tensor_Add_output_0_quantized[QLINEARADD_Add_quant_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif
// -------------------------------------------------------
//                QLINEARSIGMOID Sigmoid_1_quant
// -------------------------------------------------------

#define QSIGMOID_Sigmoid_1_quant_OUTPUT_SIZE (8)

#ifndef NCAST_SIGMOID_LUT_DEFINED
#define NCAST_SIGMOID_LUT_DEFINED
#define NCAST_SIGMOID_LUT_SIZE (256)
#define NCAST_SIGMOID_LUT_MINRANGE (-6.0f)
#define NCAST_SIGMOID_LUT_MAXRANGE (6.0f)
#define NCAST_SIGMOID_LUT_UPPER (1.0f)
#define NCAST_SIGMOID_LUT_LOWER (0.0f)
static float NCAST_SIGMOID_LUT[] = {
    0.00247262f, 0.00259145f, 0.00271598f, 0.00284647f, 0.00298322f, 0.00312651f,
    0.00327667f, 0.00343400f, 0.00359887f, 0.00377162f, 0.00395264f, 0.00414230f,
    0.00434103f, 0.00454924f, 0.00476739f, 0.00499596f, 0.00523542f, 0.00548630f,
    0.00574913f, 0.00602447f, 0.00631292f, 0.00661509f, 0.00693162f, 0.00726319f,
    0.00761049f, 0.00797427f, 0.00835529f, 0.00875435f, 0.00917230f, 0.00961000f,
    0.01006839f, 0.01054840f, 0.01105105f, 0.01157736f, 0.01212843f, 0.01270540f,
    0.01330945f, 0.01394180f, 0.01460376f, 0.01529666f, 0.01602190f, 0.01678094f,
    0.01757530f, 0.01840655f, 0.01927635f, 0.02018641f, 0.02113850f, 0.02213449f,
    0.02317629f, 0.02426591f, 0.02540543f, 0.02659699f, 0.02784285f, 0.02914532f,
    0.03050681f, 0.03192980f, 0.03341689f, 0.03497073f, 0.03659408f, 0.03828981f,
    0.04006084f, 0.04191022f, 0.04384108f, 0.04585663f, 0.04796020f, 0.05015520f,
    0.05244512f, 0.05483355f, 0.05732418f, 0.05992076f, 0.06262714f, 0.06544725f,
    0.06838509f, 0.07144472f, 0.07463028f, 0.07794595f, 0.08139597f, 0.08498462f,
    0.08871621f, 0.09259507f, 0.09662553f, 0.10081195f, 0.10515863f, 0.10966987f,
    0.11434991f, 0.11920292f, 0.12423301f, 0.12944416f, 0.13484024f, 0.14042498f,
    0.14620194f, 0.15217449f, 0.15834577f, 0.16471870f, 0.17129591f, 0.17807977f,
    0.18507228f, 0.19227514f, 0.19968963f, 0.20731665f, 0.21515667f, 0.22320969f,
    0.23147522f, 0.23995228f, 0.24863934f, 0.25753434f, 0.26663464f, 0.27593699f,
    0.28543756f, 0.29513192f, 0.30501498f, 0.31508105f, 0.32532382f, 0.33573635f,
    0.34631108f, 0.35703986f, 0.36791395f, 0.37892404f, 0.39006028f, 0.40131234f,
    0.41266938f, 0.42412013f, 0.43565293f, 0.44725578f, 0.45891636f, 0.47062211f,
    0.48236027f, 0.49411792f, 0.50588208f, 0.51763973f, 0.52937789f, 0.54108364f,
    0.55274422f, 0.56434707f, 0.57587987f, 0.58733062f, 0.59868766f, 0.60993972f,
    0.62107596f, 0.63208605f, 0.64296014f, 0.65368892f, 0.66426365f, 0.67467618f,
    0.68491895f, 0.69498502f, 0.70486808f, 0.71456244f, 0.72406301f, 0.73336536f,
    0.74246566f, 0.75136066f, 0.76004772f, 0.76852478f, 0.77679031f, 0.78484333f,
    0.79268335f, 0.80031037f, 0.80772486f, 0.81492772f, 0.82192023f, 0.82870409f,
    0.83528130f, 0.84165423f, 0.84782551f, 0.85379806f, 0.85957502f, 0.86515976f,
    0.87055584f, 0.87576699f, 0.88079708f, 0.88565009f, 0.89033013f, 0.89484137f,
    0.89918805f, 0.90337447f, 0.90740493f, 0.91128379f, 0.91501538f, 0.91860403f,
    0.92205405f, 0.92536972f, 0.92855528f, 0.93161491f, 0.93455275f, 0.93737286f,
    0.94007924f, 0.94267582f, 0.94516645f, 0.94755488f, 0.94984480f, 0.95203980f,
    0.95414337f, 0.95615892f, 0.95808978f, 0.95993916f, 0.96171019f, 0.96340592f,
    0.96502927f, 0.96658311f, 0.96807020f, 0.96949319f, 0.97085468f, 0.97215715f,
    0.97340301f, 0.97459457f, 0.97573409f, 0.97682371f, 0.97786551f, 0.97886150f,
    0.97981359f, 0.98072365f, 0.98159345f, 0.98242470f, 0.98321906f, 0.98397810f,
    0.98470334f, 0.98539624f, 0.98605820f, 0.98669055f, 0.98729460f, 0.98787157f,
    0.98842264f, 0.98894895f, 0.98945160f, 0.98993161f, 0.99039000f, 0.99082770f,
    0.99124565f, 0.99164471f, 0.99202573f, 0.99238951f, 0.99273681f, 0.99306838f,
    0.99338491f, 0.99368708f, 0.99397553f, 0.99425087f, 0.99451370f, 0.99476458f,
    0.99500404f, 0.99523261f, 0.99545076f, 0.99565897f, 0.99585770f, 0.99604736f,
    0.99622838f, 0.99640113f, 0.99656600f, 0.99672333f, 0.99687349f, 0.99701678f,
    0.99715353f, 0.99728402f, 0.99740855f, 0.99752738f
};
#endif // NCAST_SIGMOID_LUT_DEFINED

static float tensor_Sigmoid_1_quant_qsigmoid_temp[QSIGMOID_Sigmoid_1_quant_OUTPUT_SIZE];

#define CONNECTED_OUTPUT
#ifdef CONNECTED_OUTPUT
static int8_t tensor_Sigmoid_1_output_0_quantized[QSIGMOID_Sigmoid_1_quant_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif
// -------------------------------------------------------
//                QLINEARSIGMOID Sigmoid_quant
// -------------------------------------------------------

#define QSIGMOID_Sigmoid_quant_OUTPUT_SIZE (8)

#ifndef NCAST_SIGMOID_LUT_DEFINED
#define NCAST_SIGMOID_LUT_DEFINED
#define NCAST_SIGMOID_LUT_SIZE (256)
#define NCAST_SIGMOID_LUT_MINRANGE (-6.0f)
#define NCAST_SIGMOID_LUT_MAXRANGE (6.0f)
#define NCAST_SIGMOID_LUT_UPPER (1.0f)
#define NCAST_SIGMOID_LUT_LOWER (0.0f)
static float NCAST_SIGMOID_LUT[] = {
    0.00247262f, 0.00259145f, 0.00271598f, 0.00284647f, 0.00298322f, 0.00312651f,
    0.00327667f, 0.00343400f, 0.00359887f, 0.00377162f, 0.00395264f, 0.00414230f,
    0.00434103f, 0.00454924f, 0.00476739f, 0.00499596f, 0.00523542f, 0.00548630f,
    0.00574913f, 0.00602447f, 0.00631292f, 0.00661509f, 0.00693162f, 0.00726319f,
    0.00761049f, 0.00797427f, 0.00835529f, 0.00875435f, 0.00917230f, 0.00961000f,
    0.01006839f, 0.01054840f, 0.01105105f, 0.01157736f, 0.01212843f, 0.01270540f,
    0.01330945f, 0.01394180f, 0.01460376f, 0.01529666f, 0.01602190f, 0.01678094f,
    0.01757530f, 0.01840655f, 0.01927635f, 0.02018641f, 0.02113850f, 0.02213449f,
    0.02317629f, 0.02426591f, 0.02540543f, 0.02659699f, 0.02784285f, 0.02914532f,
    0.03050681f, 0.03192980f, 0.03341689f, 0.03497073f, 0.03659408f, 0.03828981f,
    0.04006084f, 0.04191022f, 0.04384108f, 0.04585663f, 0.04796020f, 0.05015520f,
    0.05244512f, 0.05483355f, 0.05732418f, 0.05992076f, 0.06262714f, 0.06544725f,
    0.06838509f, 0.07144472f, 0.07463028f, 0.07794595f, 0.08139597f, 0.08498462f,
    0.08871621f, 0.09259507f, 0.09662553f, 0.10081195f, 0.10515863f, 0.10966987f,
    0.11434991f, 0.11920292f, 0.12423301f, 0.12944416f, 0.13484024f, 0.14042498f,
    0.14620194f, 0.15217449f, 0.15834577f, 0.16471870f, 0.17129591f, 0.17807977f,
    0.18507228f, 0.19227514f, 0.19968963f, 0.20731665f, 0.21515667f, 0.22320969f,
    0.23147522f, 0.23995228f, 0.24863934f, 0.25753434f, 0.26663464f, 0.27593699f,
    0.28543756f, 0.29513192f, 0.30501498f, 0.31508105f, 0.32532382f, 0.33573635f,
    0.34631108f, 0.35703986f, 0.36791395f, 0.37892404f, 0.39006028f, 0.40131234f,
    0.41266938f, 0.42412013f, 0.43565293f, 0.44725578f, 0.45891636f, 0.47062211f,
    0.48236027f, 0.49411792f, 0.50588208f, 0.51763973f, 0.52937789f, 0.54108364f,
    0.55274422f, 0.56434707f, 0.57587987f, 0.58733062f, 0.59868766f, 0.60993972f,
    0.62107596f, 0.63208605f, 0.64296014f, 0.65368892f, 0.66426365f, 0.67467618f,
    0.68491895f, 0.69498502f, 0.70486808f, 0.71456244f, 0.72406301f, 0.73336536f,
    0.74246566f, 0.75136066f, 0.76004772f, 0.76852478f, 0.77679031f, 0.78484333f,
    0.79268335f, 0.80031037f, 0.80772486f, 0.81492772f, 0.82192023f, 0.82870409f,
    0.83528130f, 0.84165423f, 0.84782551f, 0.85379806f, 0.85957502f, 0.86515976f,
    0.87055584f, 0.87576699f, 0.88079708f, 0.88565009f, 0.89033013f, 0.89484137f,
    0.89918805f, 0.90337447f, 0.90740493f, 0.91128379f, 0.91501538f, 0.91860403f,
    0.92205405f, 0.92536972f, 0.92855528f, 0.93161491f, 0.93455275f, 0.93737286f,
    0.94007924f, 0.94267582f, 0.94516645f, 0.94755488f, 0.94984480f, 0.95203980f,
    0.95414337f, 0.95615892f, 0.95808978f, 0.95993916f, 0.96171019f, 0.96340592f,
    0.96502927f, 0.96658311f, 0.96807020f, 0.96949319f, 0.97085468f, 0.97215715f,
    0.97340301f, 0.97459457f, 0.97573409f, 0.97682371f, 0.97786551f, 0.97886150f,
    0.97981359f, 0.98072365f, 0.98159345f, 0.98242470f, 0.98321906f, 0.98397810f,
    0.98470334f, 0.98539624f, 0.98605820f, 0.98669055f, 0.98729460f, 0.98787157f,
    0.98842264f, 0.98894895f, 0.98945160f, 0.98993161f, 0.99039000f, 0.99082770f,
    0.99124565f, 0.99164471f, 0.99202573f, 0.99238951f, 0.99273681f, 0.99306838f,
    0.99338491f, 0.99368708f, 0.99397553f, 0.99425087f, 0.99451370f, 0.99476458f,
    0.99500404f, 0.99523261f, 0.99545076f, 0.99565897f, 0.99585770f, 0.99604736f,
    0.99622838f, 0.99640113f, 0.99656600f, 0.99672333f, 0.99687349f, 0.99701678f,
    0.99715353f, 0.99728402f, 0.99740855f, 0.99752738f
};
#endif // NCAST_SIGMOID_LUT_DEFINED

static float tensor_Sigmoid_quant_qsigmoid_temp[QSIGMOID_Sigmoid_quant_OUTPUT_SIZE];

#define CONNECTED_OUTPUT
#ifdef CONNECTED_OUTPUT
static int8_t tensor_Sigmoid_output_0_quantized[QSIGMOID_Sigmoid_quant_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

// -------------------------------------------------------
//                   QLINEARMUL Mul_2_quant
// -------------------------------------------------------

#define QLINEARMUL_Mul_2_quant_OUTPUT_SIZE (8)

#define CONNECTED_OUTPUT
#ifdef CONNECTED_OUTPUT
static int8_t tensor_Mul_2_output_0_quantized[QLINEARMUL_Mul_2_quant_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

// -------------------------------------------------------
//             DEQUANTIZE LINEAR Sigmoid_1_output_0_DequantizeLinear
// -------------------------------------------------------

#define DEQUANT_Sigmoid_1_output_0_DequantizeLinear_OUTPUT_SIZE (8)

#define CONNECTED_OUTPUT
#ifdef CONNECTED_OUTPUT
static float tensor_Sigmoid_1_output_0[DEQUANT_Sigmoid_1_output_0_DequantizeLinear_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

// -------------------------------------------------------
//                   QLINEARMUL Mul_quant
// -------------------------------------------------------

#define QLINEARMUL_Mul_quant_OUTPUT_SIZE (8)

#define CONNECTED_OUTPUT
#ifdef CONNECTED_OUTPUT
static int8_t tensor_Mul_output_0_quantized[QLINEARMUL_Mul_quant_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

// -------------------------------------------------------
//                   QLINEARADD Add_2_quant
// -------------------------------------------------------

#define QLINEARADD_Add_2_quant_OUTPUT_SIZE (8)

#define CONNECTED_OUTPUT
#ifdef CONNECTED_OUTPUT
static int8_t tensor_Add_2_output_0_quantized[QLINEARADD_Add_2_quant_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

// -------------------------------------------------------
//                         QLIN Sub_output_0_QuantizeLinear
// -------------------------------------------------------

#include "ncquant.h"

#define QLIN_Sub_output_0_QuantizeLinear_OUTPUT_SIZE (8)

#define CONNECTED_OUTPUT
#ifdef CONNECTED_OUTPUT
static int8_t tensor_Sub_output_0_quantized[QLIN_Sub_output_0_QuantizeLinear_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

// -------------------------------------------------------
//             DEQUANTIZE LINEAR Add_2_output_0_DequantizeLinear
// -------------------------------------------------------

#define DEQUANT_Add_2_output_0_DequantizeLinear_OUTPUT_SIZE (8)

#define CONNECTED_OUTPUT
#ifdef CONNECTED_OUTPUT
static float tensor_Add_2_output_0[DEQUANT_Add_2_output_0_DequantizeLinear_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

// -------------------------------------------------------
//                    TANH Tanh
// -------------------------------------------------------

#define TANH_Tanh_OUTPUT_SIZE (8)

#define CONNECTED_OUTPUT
#ifdef CONNECTED_OUTPUT
static float tensor_Tanh_output_0[TANH_Tanh_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

#ifndef NCAST_TANH_LUT_DEFINED
#define NCAST_TANH_LUT_DEFINED
#define NCAST_TANH_LUT_SIZE (256)
#define NCAST_TANH_LUT_MINRANGE (-2.0f)
#define NCAST_TANH_LUT_MAXRANGE (2.0f)
#define NCAST_TANH_LUT_UPPER (1.0f)
#define NCAST_TANH_LUT_LOWER (-1.0f)
static float NCAST_TANH_LUT[] = {
    -0.96402758f, -0.96290241f, -0.96174273f, -0.96054753f, -0.95931576f, -0.95804636f,
    -0.95673822f, -0.95539023f, -0.95400122f, -0.95257001f, -0.95109539f, -0.94957610f,
    -0.94801087f, -0.94639839f, -0.94473732f, -0.94302627f, -0.94126385f, -0.93944862f,
    -0.93757908f, -0.93565374f, -0.93367104f, -0.93162941f, -0.92952723f, -0.92736284f,
    -0.92513456f, -0.92284066f, -0.92047938f, -0.91804891f, -0.91554743f, -0.91297305f,
    -0.91032388f, -0.90759795f, -0.90479330f, -0.90190789f, -0.89893968f, -0.89588656f,
    -0.89274642f, -0.88951709f, -0.88619637f, -0.88278203f, -0.87927182f, -0.87566342f,
    -0.87195453f, -0.86814278f, -0.86422579f, -0.86020115f, -0.85606642f, -0.85181914f,
    -0.84745683f, -0.84297699f, -0.83837709f, -0.83365461f, -0.82880699f, -0.82383167f,
    -0.81872609f, -0.81348767f, -0.80811385f, -0.80260204f, -0.79694970f, -0.79115425f,
    -0.78521317f, -0.77912392f, -0.77288400f, -0.76649093f, -0.75994227f, -0.75323562f,
    -0.74636859f, -0.73933889f, -0.73214422f, -0.72478240f, -0.71725127f, -0.70954876f,
    -0.70167287f, -0.69362170f, -0.68539341f, -0.67698629f, -0.66839871f, -0.65962916f,
    -0.65067625f, -0.64153871f, -0.63221540f, -0.62270534f, -0.61300768f, -0.60312171f,
    -0.59304692f, -0.58278295f, -0.57232959f, -0.56168685f, -0.55085493f, -0.53983419f,
    -0.52862523f, -0.51722883f, -0.50564601f, -0.49387799f, -0.48192623f, -0.46979241f,
    -0.45747844f, -0.44498647f, -0.43231890f, -0.41947836f, -0.40646773f, -0.39329014f,
    -0.37994896f, -0.36644782f, -0.35279057f, -0.33898135f, -0.32502449f, -0.31092459f,
    -0.29668650f, -0.28231527f, -0.26781621f, -0.25319481f, -0.23845682f, -0.22360817f,
    -0.20865500f, -0.19360362f, -0.17846056f, -0.16323249f, -0.14792623f, -0.13254879f,
    -0.11710727f, -0.10160892f, -0.08606109f, -0.07047123f, -0.05484686f, -0.03919560f,
    -0.02352507f, -0.00784298f,  0.00784298f,  0.02352507f,  0.03919560f,  0.05484686f,
    0.07047123f,  0.08606109f,  0.10160892f,  0.11710727f,  0.13254879f,  0.14792623f,
    0.16323249f,  0.17846056f,  0.19360362f,  0.20865500f,  0.22360817f,  0.23845682f,
    0.25319481f,  0.26781621f,  0.28231527f,  0.29668650f,  0.31092459f,  0.32502449f,
    0.33898135f,  0.35279057f,  0.36644782f,  0.37994896f,  0.39329014f,  0.40646773f,
    0.41947836f,  0.43231890f,  0.44498647f,  0.45747844f,  0.46979241f,  0.48192623f,
    0.49387799f,  0.50564601f,  0.51722883f,  0.52862523f,  0.53983419f,  0.55085493f,
    0.56168685f,  0.57232959f,  0.58278295f,  0.59304692f,  0.60312171f,  0.61300768f,
    0.62270534f,  0.63221540f,  0.64153871f,  0.65067625f,  0.65962916f,  0.66839871f,
    0.67698629f,  0.68539341f,  0.69362170f,  0.70167287f,  0.70954876f,  0.71725127f,
    0.72478240f,  0.73214422f,  0.73933889f,  0.74636859f,  0.75323562f,  0.75994227f,
    0.76649093f,  0.77288400f,  0.77912392f,  0.78521317f,  0.79115425f,  0.79694970f,
    0.80260204f,  0.80811385f,  0.81348767f,  0.81872609f,  0.82383167f,  0.82880699f,
    0.83365461f,  0.83837709f,  0.84297699f,  0.84745683f,  0.85181914f,  0.85606642f,
    0.86020115f,  0.86422579f,  0.86814278f,  0.87195453f,  0.87566342f,  0.87927182f,
    0.88278203f,  0.88619637f,  0.88951709f,  0.89274642f,  0.89588656f,  0.89893968f,
    0.90190789f,  0.90479330f,  0.90759795f,  0.91032388f,  0.91297305f,  0.91554743f,
    0.91804891f,  0.92047938f,  0.92284066f,  0.92513456f,  0.92736284f,  0.92952723f,
    0.93162941f,  0.93367104f,  0.93565374f,  0.93757908f,  0.93944862f,  0.94126385f,
    0.94302627f,  0.94473732f,  0.94639839f,  0.94801087f,  0.94957610f,  0.95109539f,
    0.95257001f,  0.95400122f,  0.95539023f,  0.95673822f,  0.95804636f,  0.95931576f,
    0.96054753f,  0.96174273f,  0.96290241f,  0.96402758f
};
#endif // NCAST_TANH_LUT_DEFINED

// -------------------------------------------------------
//                         QLIN Tanh_output_0_QuantizeLinear
// -------------------------------------------------------

#include "ncquant.h"

#define QLIN_Tanh_output_0_QuantizeLinear_OUTPUT_SIZE (8)

#define CONNECTED_OUTPUT
#ifdef CONNECTED_OUTPUT
static int8_t tensor_Tanh_output_0_quantized[QLIN_Tanh_output_0_QuantizeLinear_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

// -------------------------------------------------------
//                   QLINEARMUL Mul_1_quant
// -------------------------------------------------------

#define QLINEARMUL_Mul_1_quant_OUTPUT_SIZE (8)

#define CONNECTED_OUTPUT
#ifdef CONNECTED_OUTPUT
static int8_t tensor_Mul_1_output_0_quantized[QLINEARMUL_Mul_1_quant_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

// -------------------------------------------------------
//                   QLINEARADD Add_3_quant
// -------------------------------------------------------

#define QLINEARADD_Add_3_quant_OUTPUT_SIZE (8)

#define CONNECTED_OUTPUT
#ifdef CONNECTED_OUTPUT
static int8_t tensor_Add_3_output_0_quantized[QLINEARADD_Add_3_quant_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

// -------------------------------------------------------
//                         QGEMM fcGemm_quant
// -------------------------------------------------------

#define QGEMM_fcGemm_quant_OUTPUT_SIZE (2)
#define QGEMM_fcGemm_quant_INPUT_SIZE (8)

#define CONNECTED_OUTPUT
#ifdef CONNECTED_OUTPUT
static int8_t tensor_output_quantized[QGEMM_fcGemm_quant_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

// -------------------------------------------------------
//             DEQUANTIZE LINEAR output_DequantizeLinear
// -------------------------------------------------------

#define DEQUANT_output_DequantizeLinear_OUTPUT_SIZE (2)


#ifdef CONNECTED_OUTPUT
static float tensor_output[DEQUANT_output_DequantizeLinear_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif
void run_inference(float32_t* tensor_onnxGemm_0, float32_t* tensor_onnxGemm_1, float32_t* tensor_output) {







// -------------------------------------------------------
//                         QLIN onnxGemm_0_QuantizeLinear
// -------------------------------------------------------

NCAST_QUANT8(tensor_onnxGemm_0, tensor_onnxGemm_0_quantized, QLIN_onnxGemm_0_QuantizeLinear_OUTPUT_SIZE, tensor_onnxGemm_0_scale[0], tensor_onnxGemm_0_zero_point[0])

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT onnxGemm_0_QuantizeLinear -----------------\n");
for(int i=0; i<QLIN_onnxGemm_0_QuantizeLinear_OUTPUT_SIZE; i++) {
    printf("%d ", tensor_onnxGemm_0_quantized[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif



















// -------------------------------------------------------
//                         QLIN onnxGemm_1_QuantizeLinear
// -------------------------------------------------------

NCAST_QUANT8(tensor_onnxGemm_1, tensor_onnxGemm_1_quantized, QLIN_onnxGemm_1_QuantizeLinear_OUTPUT_SIZE, tensor_onnxGemm_1_scale[0], tensor_onnxGemm_1_zero_point[0])

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT onnxGemm_1_QuantizeLinear -----------------\n");
for(int i=0; i<QLIN_onnxGemm_1_QuantizeLinear_OUTPUT_SIZE; i++) {
    printf("%d ", tensor_onnxGemm_1_quantized[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif


















// QGEMM OPERATOR W_izGemm_quant

{
    int32_t temp;
    for(int i=0; i<QGEMM_W_izGemm_quant_OUTPUT_SIZE; i++) {
        temp = 0;
        for(int j=0; j<QGEMM_W_izGemm_quant_INPUT_SIZE; j++) {
            temp += (int32_t)tensor_W_izweight_quantized[i*QGEMM_W_izGemm_quant_INPUT_SIZE+j] * ((int32_t)tensor_onnxGemm_0_quantized[j]-(int32_t)tensor_onnxGemm_0_zero_point[0]);        
        }
        temp += tensor_ortshared_1_1_8_1_token_67_quantized[i];

        float sw = tensor_W_izweight_scale[0];
        float sx = tensor_onnxGemm_0_scale[0];
        float sy = tensor_W_izGemm_output_0_scale[0];
        float swxy = (sw*sx)/sy;

        int32_t res_i = ((int32_t)NCAST_ROUND(swxy * temp)) + (tensor_W_izGemm_output_0_zero_point[0]);

        NCAST_CLIP_INT8(res_i)
        tensor_W_izGemm_output_0_quantized[i] = (int8_t)res_i;
    }
}



#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT W_izGemm_quant -----------------\n");
for(int i=0; i<QGEMM_W_izGemm_quant_OUTPUT_SIZE; i++) {
    printf("%d ", tensor_W_izGemm_output_0_quantized[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// QGEMM OPERATOR W_hzGemm_quant

{
    int32_t temp;
    for(int i=0; i<QGEMM_W_hzGemm_quant_OUTPUT_SIZE; i++) {
        temp = 0;
        for(int j=0; j<QGEMM_W_hzGemm_quant_INPUT_SIZE; j++) {
            temp += (int32_t)tensor_W_hzweight_quantized[i*QGEMM_W_hzGemm_quant_INPUT_SIZE+j] * ((int32_t)tensor_onnxGemm_1_quantized[j]-(int32_t)tensor_onnxGemm_1_zero_point[0]);        
        }
        temp += tensor_ortshared_1_1_8_0_token_65_quantized[i];

        float sw = tensor_W_hzweight_scale[0];
        float sx = tensor_onnxGemm_1_scale[0];
        float sy = tensor_W_hzGemm_output_0_scale[0];
        float swxy = (sw*sx)/sy;

        int32_t res_i = ((int32_t)NCAST_ROUND(swxy * temp)) + (tensor_W_hzGemm_output_0_zero_point[0]);

        NCAST_CLIP_INT8(res_i)
        tensor_W_hzGemm_output_0_quantized[i] = (int8_t)res_i;
    }
}



#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT W_hzGemm_quant -----------------\n");
for(int i=0; i<QGEMM_W_hzGemm_quant_OUTPUT_SIZE; i++) {
    printf("%d ", tensor_W_hzGemm_output_0_quantized[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif


// QGEMM OPERATOR W_irGemm_quant

{
    int32_t temp;
    for(int i=0; i<QGEMM_W_irGemm_quant_OUTPUT_SIZE; i++) {
        temp = 0;
        for(int j=0; j<QGEMM_W_irGemm_quant_INPUT_SIZE; j++) {
            temp += (int32_t)tensor_W_irweight_quantized[i*QGEMM_W_irGemm_quant_INPUT_SIZE+j] * ((int32_t)tensor_onnxGemm_0_quantized[j]-(int32_t)tensor_onnxGemm_0_zero_point[0]);        
        }
        temp += tensor_ortshared_1_1_8_2_token_68_quantized[i];

        float sw = tensor_W_irweight_scale[0];
        float sx = tensor_onnxGemm_0_scale[0];
        float sy = tensor_W_irGemm_output_0_scale[0];
        float swxy = (sw*sx)/sy;

        int32_t res_i = ((int32_t)NCAST_ROUND(swxy * temp)) + (tensor_W_irGemm_output_0_zero_point[0]);

        NCAST_CLIP_INT8(res_i)
        tensor_W_irGemm_output_0_quantized[i] = (int8_t)res_i;
    }
}



#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT W_irGemm_quant -----------------\n");
for(int i=0; i<QGEMM_W_irGemm_quant_OUTPUT_SIZE; i++) {
    printf("%d ", tensor_W_irGemm_output_0_quantized[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// QGEMM OPERATOR W_hrGemm_quant

{
    int32_t temp;
    for(int i=0; i<QGEMM_W_hrGemm_quant_OUTPUT_SIZE; i++) {
        temp = 0;
        for(int j=0; j<QGEMM_W_hrGemm_quant_INPUT_SIZE; j++) {
            temp += (int32_t)tensor_W_hrweight_quantized[i*QGEMM_W_hrGemm_quant_INPUT_SIZE+j] * ((int32_t)tensor_onnxGemm_1_quantized[j]-(int32_t)tensor_onnxGemm_1_zero_point[0]);        
        }
        temp += tensor_ortshared_1_1_8_4_token_70_quantized[i];

        float sw = tensor_W_hrweight_scale[0];
        float sx = tensor_onnxGemm_1_scale[0];
        float sy = tensor_W_hrGemm_output_0_scale[0];
        float swxy = (sw*sx)/sy;

        int32_t res_i = ((int32_t)NCAST_ROUND(swxy * temp)) + (tensor_W_hrGemm_output_0_zero_point[0]);

        NCAST_CLIP_INT8(res_i)
        tensor_W_hrGemm_output_0_quantized[i] = (int8_t)res_i;
    }
}



#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT W_hrGemm_quant -----------------\n");
for(int i=0; i<QGEMM_W_hrGemm_quant_OUTPUT_SIZE; i++) {
    printf("%d ", tensor_W_hrGemm_output_0_quantized[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif



// QLINEARADD OPERATOR Add_1_quant

{
float32_t say = tensor_W_izGemm_output_0_scale[0] / tensor_Add_1_output_0_scale[0];
float32_t sby = tensor_W_hzGemm_output_0_scale[0] / tensor_Add_1_output_0_scale[0];

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<8; i1++) {

int32_t qaz_i = (int32_t)tensor_W_izGemm_output_0_quantized[i1*1] - (int32_t)tensor_W_izGemm_output_0_zero_point[0];
int32_t qbz_i = (int32_t)tensor_W_hzGemm_output_0_quantized[i1*1] - (int32_t)tensor_W_hzGemm_output_0_zero_point[0];
int32_t res_i = ((int32_t)NCAST_ROUND(say*qaz_i)) + ((int32_t)NCAST_ROUND(sby*qbz_i)) + (int32_t)tensor_Add_1_output_0_zero_point[0];
NCAST_CLIP_INT8(res_i)
tensor_Add_1_output_0_quantized[i1*1] = (int8_t)res_i;
}
}

}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Add_1_quant -----------------\n");
for(int i=0; i<QLINEARADD_Add_1_quant_OUTPUT_SIZE; i++) {
    printf("%d ", tensor_Add_1_output_0_quantized[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif




// QLINEARADD OPERATOR Add_quant

{
float32_t say = tensor_W_irGemm_output_0_scale[0] / tensor_Add_output_0_scale[0];
float32_t sby = tensor_W_hrGemm_output_0_scale[0] / tensor_Add_output_0_scale[0];

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<8; i1++) {

int32_t qaz_i = (int32_t)tensor_W_irGemm_output_0_quantized[i1*1] - (int32_t)tensor_W_irGemm_output_0_zero_point[0];
int32_t qbz_i = (int32_t)tensor_W_hrGemm_output_0_quantized[i1*1] - (int32_t)tensor_W_hrGemm_output_0_zero_point[0];
int32_t res_i = ((int32_t)NCAST_ROUND(say*qaz_i)) + ((int32_t)NCAST_ROUND(sby*qbz_i)) + (int32_t)tensor_Add_output_0_zero_point[0];
NCAST_CLIP_INT8(res_i)
tensor_Add_output_0_quantized[i1*1] = (int8_t)res_i;
}
}

}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Add_quant -----------------\n");
for(int i=0; i<QLINEARADD_Add_quant_OUTPUT_SIZE; i++) {
    printf("%d ", tensor_Add_output_0_quantized[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif




// QLINEARSIGMOID Sigmoid_1_quant

NCAST_DQUANT8(tensor_Add_1_output_0_quantized, tensor_Sigmoid_1_quant_qsigmoid_temp, QSIGMOID_Sigmoid_1_quant_OUTPUT_SIZE, tensor_Add_1_output_0_scale[0], tensor_Add_1_output_0_zero_point[0])

for(int i=0; i<QSIGMOID_Sigmoid_1_quant_OUTPUT_SIZE; i++) {
    float y;
    NCAST_ACCESS_LUT(NCAST_SIGMOID_LUT, tensor_Sigmoid_1_quant_qsigmoid_temp[i], y, NCAST_SIGMOID_LUT_MINRANGE, NCAST_SIGMOID_LUT_MAXRANGE, NCAST_SIGMOID_LUT_UPPER, NCAST_SIGMOID_LUT_LOWER, NCAST_SIGMOID_LUT_SIZE)
    tensor_Sigmoid_1_quant_qsigmoid_temp[i] = y;
}

NCAST_QUANT8(tensor_Sigmoid_1_quant_qsigmoid_temp, tensor_Sigmoid_1_output_0_quantized, QSIGMOID_Sigmoid_1_quant_OUTPUT_SIZE, tensor_Sigmoid_1_output_0_scale[0], tensor_Sigmoid_1_output_0_zero_point[0])

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Sigmoid_1_quant -----------------\n");
for(int i=0; i<QSIGMOID_Sigmoid_1_quant_OUTPUT_SIZE; i++) {
    printf("%d ", tensor_Sigmoid_1_output_0_quantized[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif




// QLINEARSIGMOID Sigmoid_quant

NCAST_DQUANT8(tensor_Add_output_0_quantized, tensor_Sigmoid_quant_qsigmoid_temp, QSIGMOID_Sigmoid_quant_OUTPUT_SIZE, tensor_Add_output_0_scale[0], tensor_Add_output_0_zero_point[0])

for(int i=0; i<QSIGMOID_Sigmoid_quant_OUTPUT_SIZE; i++) {
    float y;
    NCAST_ACCESS_LUT(NCAST_SIGMOID_LUT, tensor_Sigmoid_quant_qsigmoid_temp[i], y, NCAST_SIGMOID_LUT_MINRANGE, NCAST_SIGMOID_LUT_MAXRANGE, NCAST_SIGMOID_LUT_UPPER, NCAST_SIGMOID_LUT_LOWER, NCAST_SIGMOID_LUT_SIZE)
    tensor_Sigmoid_quant_qsigmoid_temp[i] = y;
}

NCAST_QUANT8(tensor_Sigmoid_quant_qsigmoid_temp, tensor_Sigmoid_output_0_quantized, QSIGMOID_Sigmoid_quant_OUTPUT_SIZE, tensor_Sigmoid_output_0_scale[0], tensor_Sigmoid_output_0_zero_point[0])

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Sigmoid_quant -----------------\n");
for(int i=0; i<QSIGMOID_Sigmoid_quant_OUTPUT_SIZE; i++) {
    printf("%d ", tensor_Sigmoid_output_0_quantized[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif

// QGEMM OPERATOR W_hnGemm_quant

{
    int32_t temp;
    for(int i=0; i<QGEMM_W_hnGemm_quant_OUTPUT_SIZE; i++) {
        temp = 0;
        for(int j=0; j<QGEMM_W_hnGemm_quant_INPUT_SIZE; j++) {
            temp += (int32_t)tensor_W_hnweight_quantized[i*QGEMM_W_hnGemm_quant_INPUT_SIZE+j] * ((int32_t)tensor_onnxGemm_1_quantized[j]-(int32_t)tensor_onnxGemm_1_zero_point[0]);        
        }
        temp += tensor_ortshared_1_1_8_3_token_69_quantized[i];

        float sw = tensor_W_hnweight_scale[0];
        float sx = tensor_onnxGemm_1_scale[0];
        float sy = tensor_W_hnGemm_output_0_scale[0];
        float swxy = (sw*sx)/sy;

        int32_t res_i = ((int32_t)NCAST_ROUND(swxy * temp)) + (tensor_W_hnGemm_output_0_zero_point[0]);

        NCAST_CLIP_INT8(res_i)
        tensor_W_hnGemm_output_0_quantized[i] = (int8_t)res_i;
    }
}



#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT W_hnGemm_quant -----------------\n");
for(int i=0; i<QGEMM_W_hnGemm_quant_OUTPUT_SIZE; i++) {
    printf("%d ", tensor_W_hnGemm_output_0_quantized[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif




// -------------------------------------------------------
//             DEQUANTIZE LINEAR Sigmoid_1_output_0_DequantizeLinear
// -------------------------------------------------------

NCAST_DQUANT8(tensor_Sigmoid_1_output_0_quantized, tensor_Sigmoid_1_output_0, DEQUANT_Sigmoid_1_output_0_DequantizeLinear_OUTPUT_SIZE, tensor_Sigmoid_1_output_0_scale[0], tensor_Sigmoid_1_output_0_zero_point[0])

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Sigmoid_1_output_0_DequantizeLinear -----------------\n");
for(int i=0; i<8; i++) {
    printf("%f ", tensor_Sigmoid_1_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// QGEMM OPERATOR W_inGemm_quant

{
    int32_t temp;
    for(int i=0; i<QGEMM_W_inGemm_quant_OUTPUT_SIZE; i++) {
        temp = 0;
        for(int j=0; j<QGEMM_W_inGemm_quant_INPUT_SIZE; j++) {
            temp += (int32_t)tensor_W_inweight_quantized[i*QGEMM_W_inGemm_quant_INPUT_SIZE+j] * ((int32_t)tensor_onnxGemm_0_quantized[j]-(int32_t)tensor_onnxGemm_0_zero_point[0]);        
        }
        temp += tensor_ortshared_1_1_8_5_token_71_quantized[i];

        float sw = tensor_W_inweight_scale[0];
        float sx = tensor_onnxGemm_0_scale[0];
        float sy = tensor_W_inGemm_output_0_scale[0];
        float swxy = (sw*sx)/sy;

        int32_t res_i = ((int32_t)NCAST_ROUND(swxy * temp)) + (tensor_W_inGemm_output_0_zero_point[0]);

        NCAST_CLIP_INT8(res_i)
        tensor_W_inGemm_output_0_quantized[i] = (int8_t)res_i;
    }
}



#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT W_inGemm_quant -----------------\n");
for(int i=0; i<QGEMM_W_inGemm_quant_OUTPUT_SIZE; i++) {
    printf("%d ", tensor_W_inGemm_output_0_quantized[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif

// QLINEARMUL OPERATOR Mul_quant

{
float32_t saby = (tensor_Sigmoid_output_0_scale[0] * tensor_W_hnGemm_output_0_scale[0]) / tensor_Mul_output_0_scale[0];

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<8; i1++) {

int32_t qaz_i = (int32_t)tensor_Sigmoid_output_0_quantized[i1*1] - (int32_t)tensor_Sigmoid_output_0_zero_point[0];
int32_t qbz_i = (int32_t)tensor_W_hnGemm_output_0_quantized[i1*1] - (int32_t)tensor_W_hnGemm_output_0_zero_point[0];
int32_t res_i = ((int32_t)NCAST_ROUND(saby * qaz_i * qbz_i)) + (int32_t)tensor_Mul_output_0_zero_point[0];
NCAST_CLIP_INT8(res_i)
tensor_Mul_output_0_quantized[i1*1] = (int8_t)res_i;
}
}

}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Mul_quant -----------------\n");
for(int i=0; i<QLINEARMUL_Mul_quant_OUTPUT_SIZE; i++) {
    printf("%d ", tensor_Mul_output_0_quantized[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif



// ELEMENT WISE SUBTRACTION Sub

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_Sub_output_0[8];
#undef CONNECTED_OUTPUT
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<8; i1++) {

tensor_Sub_output_0[i1*1] = tensor_ortshared_1_0_1_0_token_66[0] - tensor_Sigmoid_1_output_0[i1*1];
}
}


#ifdef COMPILER_BENCHMARK
BENCHMARK("tensor_Sub", 8)
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Sub -----------------\n");
for(int i=0; i<8; i++) {
    printf("%f ", tensor_Sub_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif



// QLINEARADD OPERATOR Add_2_quant

{
float32_t say = tensor_W_inGemm_output_0_scale[0] / tensor_Add_2_output_0_scale[0];
float32_t sby = tensor_Mul_output_0_scale[0] / tensor_Add_2_output_0_scale[0];

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<8; i1++) {

int32_t qaz_i = (int32_t)tensor_W_inGemm_output_0_quantized[i1*1] - (int32_t)tensor_W_inGemm_output_0_zero_point[0];
int32_t qbz_i = (int32_t)tensor_Mul_output_0_quantized[i1*1] - (int32_t)tensor_Mul_output_0_zero_point[0];
int32_t res_i = ((int32_t)NCAST_ROUND(say*qaz_i)) + ((int32_t)NCAST_ROUND(sby*qbz_i)) + (int32_t)tensor_Add_2_output_0_zero_point[0];
NCAST_CLIP_INT8(res_i)
tensor_Add_2_output_0_quantized[i1*1] = (int8_t)res_i;
}
}

}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Add_2_quant -----------------\n");
for(int i=0; i<QLINEARADD_Add_2_quant_OUTPUT_SIZE; i++) {
    printf("%d ", tensor_Add_2_output_0_quantized[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif


// -------------------------------------------------------
//             DEQUANTIZE LINEAR Add_2_output_0_DequantizeLinear
// -------------------------------------------------------

NCAST_DQUANT8(tensor_Add_2_output_0_quantized, tensor_Add_2_output_0, DEQUANT_Add_2_output_0_DequantizeLinear_OUTPUT_SIZE, tensor_Add_2_output_0_scale[0], tensor_Add_2_output_0_zero_point[0])

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Add_2_output_0_DequantizeLinear -----------------\n");
for(int i=0; i<8; i++) {
    printf("%f ", tensor_Add_2_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// TANH OPERATOR Tanh

for(int i=0; i<TANH_Tanh_OUTPUT_SIZE; i++) {
    float y;
    NCAST_ACCESS_LUT(NCAST_TANH_LUT, tensor_Add_2_output_0[i], y, NCAST_TANH_LUT_MINRANGE, NCAST_TANH_LUT_MAXRANGE, NCAST_TANH_LUT_UPPER, NCAST_TANH_LUT_LOWER, NCAST_TANH_LUT_SIZE)
    tensor_Tanh_output_0[i] = y;
}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Tanh -----------------\n");
for(int i=0; i<TANH_Tanh_OUTPUT_SIZE; i++) {
    printf("%f ", tensor_Tanh_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif



// -------------------------------------------------------
//                         QLIN Sub_output_0_QuantizeLinear
// -------------------------------------------------------

NCAST_QUANT8(tensor_Sub_output_0, tensor_Sub_output_0_quantized, QLIN_Sub_output_0_QuantizeLinear_OUTPUT_SIZE, tensor_Sub_output_0_scale[0], tensor_Sub_output_0_zero_point[0])

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Sub_output_0_QuantizeLinear -----------------\n");
for(int i=0; i<QLIN_Sub_output_0_QuantizeLinear_OUTPUT_SIZE; i++) {
    printf("%d ", tensor_Sub_output_0_quantized[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif

// -------------------------------------------------------
//                         QLIN Tanh_output_0_QuantizeLinear
// -------------------------------------------------------

NCAST_QUANT8(tensor_Tanh_output_0, tensor_Tanh_output_0_quantized, QLIN_Tanh_output_0_QuantizeLinear_OUTPUT_SIZE, tensor_Tanh_output_0_scale[0], tensor_Tanh_output_0_zero_point[0])

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Tanh_output_0_QuantizeLinear -----------------\n");
for(int i=0; i<QLIN_Tanh_output_0_QuantizeLinear_OUTPUT_SIZE; i++) {
    printf("%d ", tensor_Tanh_output_0_quantized[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif



// QLINEARMUL OPERATOR Mul_1_quant

{
float32_t saby = (tensor_Sub_output_0_scale[0] * tensor_Tanh_output_0_scale[0]) / tensor_Mul_1_output_0_scale[0];

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<8; i1++) {

int32_t qaz_i = (int32_t)tensor_Sub_output_0_quantized[i1*1] - (int32_t)tensor_Sub_output_0_zero_point[0];
int32_t qbz_i = (int32_t)tensor_Tanh_output_0_quantized[i1*1] - (int32_t)tensor_Tanh_output_0_zero_point[0];
int32_t res_i = ((int32_t)NCAST_ROUND(saby * qaz_i * qbz_i)) + (int32_t)tensor_Mul_1_output_0_zero_point[0];
NCAST_CLIP_INT8(res_i)
tensor_Mul_1_output_0_quantized[i1*1] = (int8_t)res_i;
}
}

}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Mul_1_quant -----------------\n");
for(int i=0; i<QLINEARMUL_Mul_1_quant_OUTPUT_SIZE; i++) {
    printf("%d ", tensor_Mul_1_output_0_quantized[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif


// QLINEARMUL OPERATOR Mul_2_quant

{
float32_t saby = (tensor_Sigmoid_1_output_0_scale[0] * tensor_onnxGemm_1_scale[0]) / tensor_Mul_2_output_0_scale[0];

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<8; i1++) {

int32_t qaz_i = (int32_t)tensor_Sigmoid_1_output_0_quantized[i1*1] - (int32_t)tensor_Sigmoid_1_output_0_zero_point[0];
int32_t qbz_i = (int32_t)tensor_onnxGemm_1_quantized[i1*1] - (int32_t)tensor_onnxGemm_1_zero_point[0];
int32_t res_i = ((int32_t)NCAST_ROUND(saby * qaz_i * qbz_i)) + (int32_t)tensor_Mul_2_output_0_zero_point[0];
NCAST_CLIP_INT8(res_i)
tensor_Mul_2_output_0_quantized[i1*1] = (int8_t)res_i;
}
}

}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Mul_2_quant -----------------\n");
for(int i=0; i<QLINEARMUL_Mul_2_quant_OUTPUT_SIZE; i++) {
    printf("%d ", tensor_Mul_2_output_0_quantized[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif




// QLINEARADD OPERATOR Add_3_quant

{
float32_t say = tensor_Mul_1_output_0_scale[0] / tensor_Add_3_output_0_scale[0];
float32_t sby = tensor_Mul_2_output_0_scale[0] / tensor_Add_3_output_0_scale[0];

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<8; i1++) {

int32_t qaz_i = (int32_t)tensor_Mul_1_output_0_quantized[i1*1] - (int32_t)tensor_Mul_1_output_0_zero_point[0];
int32_t qbz_i = (int32_t)tensor_Mul_2_output_0_quantized[i1*1] - (int32_t)tensor_Mul_2_output_0_zero_point[0];
int32_t res_i = ((int32_t)NCAST_ROUND(say*qaz_i)) + ((int32_t)NCAST_ROUND(sby*qbz_i)) + (int32_t)tensor_Add_3_output_0_zero_point[0];
NCAST_CLIP_INT8(res_i)
tensor_Add_3_output_0_quantized[i1*1] = (int8_t)res_i;
}
}

}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Add_3_quant -----------------\n");
for(int i=0; i<QLINEARADD_Add_3_quant_OUTPUT_SIZE; i++) {
    printf("%d ", tensor_Add_3_output_0_quantized[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif







// QGEMM OPERATOR fcGemm_quant

{
    int32_t temp;
    for(int i=0; i<QGEMM_fcGemm_quant_OUTPUT_SIZE; i++) {
        temp = 0;
        for(int j=0; j<QGEMM_fcGemm_quant_INPUT_SIZE; j++) {
            temp += (int32_t)tensor_fcweight_quantized[i*QGEMM_fcGemm_quant_INPUT_SIZE+j] * ((int32_t)tensor_Add_3_output_0_quantized[j]-(int32_t)tensor_Add_3_output_0_zero_point[0]);        
        }
        temp += tensor_ortshared_1_1_2_0_token_64_quantized[i];

        float sw = tensor_fcweight_scale[0];
        float sx = tensor_Add_3_output_0_scale[0];
        float sy = tensor_output_scale[0];
        float swxy = (sw*sx)/sy;

        int32_t res_i = ((int32_t)NCAST_ROUND(swxy * temp)) + (tensor_output_zero_point[0]);

        NCAST_CLIP_INT8(res_i)
        tensor_output_quantized[i] = (int8_t)res_i;
    }
}



#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT fcGemm_quant -----------------\n");
for(int i=0; i<QGEMM_fcGemm_quant_OUTPUT_SIZE; i++) {
    printf("%d ", tensor_output_quantized[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif

// -------------------------------------------------------
//             DEQUANTIZE LINEAR output_DequantizeLinear
// -------------------------------------------------------

NCAST_DQUANT8(tensor_output_quantized, tensor_output, DEQUANT_output_DequantizeLinear_OUTPUT_SIZE, tensor_output_scale[0], tensor_output_zero_point[0])

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT output_DequantizeLinear -----------------\n");
for(int i=0; i<2; i++) {
    printf("%f ", tensor_output[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
}