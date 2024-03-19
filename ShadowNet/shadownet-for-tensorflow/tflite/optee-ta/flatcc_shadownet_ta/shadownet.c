#include<stdio.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<stdbool.h>
#include <float.h>
#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>
#include <math_ta.h>
#include <shadownet.h>
#include <hello_world_ta.h>
#include <test_read_tflite_generated.h>

#undef ns
#define ns(x) FLATBUFFERS_WRAP_NAMESPACE(tflite, x)

// LinearTransform operator input tensors index definition
#define LT_INPUT_IDX            0
#define LT_WEIGHT1_IDX          1
#define LT_WEIGHT2_IDX          2 
#define LT_WEIGHT3_IDX          3 

// input tensor shape index (batch, h, w, c)
#define BATCH_IDX               0 
#define HEIGHT_IDX              1 
#define WIDTH_IDX               2 
#define CHN_IDX                 3 

// output tensor c is represented as units
#define UNIT_IDX                1 

#ifdef USE_MOBILENETS
// each layer input/output max buffer size
#define SN_MAX_BUF_SIZE (112*112*76*4)
#else
#define SN_MAX_BUF_SIZE (32*32*32*4) // RESNET
#endif

// optimize MUL + ADD operations
#define OPTIMIZE_MUL_ADD

// RESNET
#define RESNET_PNUM             45 // 45 convs
#define RESNET_MAX_OPS          9//  9 tee ops
#define RESNET_PLAN_NODE_FILL(plan_id, node_id, opid,opcode, rb, wb)     \
    resnet_plans[plan_id].nodes[node_id].op_id = opid;          \
    resnet_plans[plan_id].nodes[node_id].op_code = opcode;      \
    resnet_plans[plan_id].nodes[node_id].read_buf = rb;         \
    resnet_plans[plan_id].nodes[node_id].write_buf = wb

#define RESNET_PLAN_SET_NODE_NUM(plan_id, nodenum)              \
    resnet_plans[plan_id].node_num = nodenum

typedef struct execution_node {
    unsigned op_id;
    unsigned op_code;
    bool read_buf; // the second input node is from tempbuf
    bool write_buf; // this operator write to tempbuf and outbuf 
} enode_t;

typedef struct execution_plan {
    enode_t nodes[RESNET_MAX_OPS];
    unsigned node_num;
} eplan_t;

eplan_t resnet_plans[RESNET_PNUM] = {0};

// Below are global data structures about this model
// This is a read-only reference to a flatbuffer encoded struct. 
ns(Model_table_t) model;
ns(SubGraph_vec_t) model_subgraphs;
ns(SubGraph_table_t) model_sub0;
ns(Operator_vec_t) model_operators;
ns(OperatorCode_vec_t) model_opcodes;
ns(Tensor_vec_t) model_tensors;
ns(Buffer_vec_t) model_buffers;

// two output buffers, rotate them for shadownet layers to save memory
float *outbuf1, *outbuf2, *tempbuf;

static inline float relu6_activate(float x){return (x < 0.) ? 0 : (6.0 < x) ? 6.0: x;}
static inline float relu_activate(float x){return (x < 0.) ? 0 : x;}

enum ActivationFunctionType {
  ACT_NONE = 0,
  ACT_RELU = 1,
  ACT_RELU_N1_TO_1 = 2,
  ACT_RELU6 = 3,
  ACT_TANH = 4,
  ACT_SIGN_BIT = 5,
};

enum BuiltinOperator {
  ADD = 0,
  AVERAGE_POOL_2D = 1,
  CONCATENATION = 2,
  CONV_2D = 3,
  DEPTHWISE_CONV_2D = 4,
  DEPTH_TO_SPACE = 5,
  DEQUANTIZE = 6,
  EMBEDDING_LOOKUP = 7,
  FLOOR = 8,
  FULLY_CONNECTED = 9,
  HASHTABLE_LOOKUP = 10,
  L2_NORMALIZATION = 11,
  L2_POOL_2D = 12,
  LOCAL_RESPONSE_NORMALIZATION = 13,
  LOGISTIC = 14,
  LSH_PROJECTION = 15,
  LSTM = 16,
  MAX_POOL_2D = 17,
  MUL = 18,
  RELU = 19,
  // NOTE(aselle): RELU_N1_TO_1 used to be called RELU1, but it was renamed
  // since different model developers use RELU1 in different ways. Never
  // create another op called RELU1.
  RELU_N1_TO_1 = 20,
  RELU6 = 21,
  RESHAPE = 22,
  RESIZE_BILINEAR = 23,
  RNN = 24,
  SOFTMAX = 25,
  SPACE_TO_DEPTH = 26,
  SVDF = 27,
  TANH = 28,
  // TODO(aselle): Consider rename to CONCATENATE_EMBEDDINGS
  CONCAT_EMBEDDINGS = 29,
  SKIP_GRAM = 30,
  CALL = 31,
  CUSTOM = 32,
  EMBEDDING_LOOKUP_SPARSE = 33,
  PAD = 34,
  UNIDIRECTIONAL_SEQUENCE_RNN = 35,
  GATHER = 36,
  BATCH_TO_SPACE_ND = 37,
  SPACE_TO_BATCH_ND = 38,
  TRANSPOSE = 39,
  MEAN = 40,
  SUB = 41,
  DIV = 42,
  SQUEEZE = 43,
  UNIDIRECTIONAL_SEQUENCE_LSTM = 44,
  STRIDED_SLICE = 45,
  BIDIRECTIONAL_SEQUENCE_RNN = 46,
  EXP = 47,
  TOPK_V2 = 48,
  SPLIT = 49,
  LOG_SOFTMAX = 50,
  // DELEGATE is a special op type for the operations which are delegated to
  // other backends.
  // WARNING: Experimental interface, subject to change
  DELEGATE = 51,
  BIDIRECTIONAL_SEQUENCE_LSTM = 52,
  CAST = 53,
  PRELU = 54,
  MAXIMUM = 55,
  ARG_MAX = 56,
  MINIMUM = 57,
  LESS = 58,
  NEG = 59,
  PADV2 = 60,
  GREATER = 61,
  GREATER_EQUAL = 62,
  LESS_EQUAL = 63,
  SELECT = 64,
  SLICE = 65,
  SIN = 66,
  TRANSPOSE_CONV = 67,
  SPARSE_TO_DENSE = 68,
  TILE = 69,
  EXPAND_DIMS = 70,
  EQUAL = 71,
  NOT_EQUAL = 72,
  LOG = 73,
  SUM = 74,
  SQRT = 75,
  RSQRT = 76,
  SHAPE = 77,
  POW = 78,
  ARG_MIN = 79,
  FAKE_QUANT = 80,
  REDUCE_PROD = 81,
  REDUCE_MAX = 82,
  PACK = 83,
  LOGICAL_OR = 84,
  ONE_HOT = 85,
  LOGICAL_AND = 86,
  LOGICAL_NOT = 87,
  UNPACK = 88,
  REDUCE_MIN = 89,
  FLOOR_DIV = 90,
  REDUCE_ANY = 91,
  SQUARE = 92,
  ZEROS_LIKE = 93,
  FILL = 94,
  FLOOR_MOD = 95,
  RANGE = 96,
  RESIZE_NEAREST_NEIGHBOR = 97,
  LEAKY_RELU = 98,
  SQUARED_DIFFERENCE = 99,
  MIRROR_PAD = 100,
  ABS = 101,
  SPLIT_V = 102,
  UNIQUE = 103,
  CEIL = 104,
  REVERSE_V2 = 105,
  ADD_N = 106,
  GATHER_ND = 107,
  COS = 108,
  WHERE = 109,
  RANK = 110,
  ELU = 111,
  REVERSE_SEQUENCE = 112,
  MATRIX_DIAG = 113,
  QUANTIZE = 114,
  MATRIX_SET_DIAG = 115,
  ROUND = 116,
  HARD_SWISH = 117,
  IF = 118,
  WHILE = 119,
  NON_MAX_SUPPRESSION_V4 = 120,
  NON_MAX_SUPPRESSION_V5 = 121,
  SCATTER_ND = 122,
  SELECT_V2 = 123,
  DENSIFY = 124,
  SEGMENT_SUM = 125
};

// TODO
// this execution plan is hand-tuned, and it assumes
// some parallel ops to be executed in certain sequence
// To remove such dependency, one idea is to introduce
// a tee_sync operation to create a actual data-flow merge
// inside tee, so that tensorflow will schedule the ops
// as we want.
void init_resnet_plan() {
	RESNET_PLAN_NODE_FILL(0, 0, 1, 32, false, false);
	RESNET_PLAN_NODE_FILL(0, 1, 2, 18, false, false);
	RESNET_PLAN_NODE_FILL(0, 2, 3, 0, false, true);
	RESNET_PLAN_NODE_FILL(0, 3, 4, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(0, 4);

	RESNET_PLAN_NODE_FILL(1, 0, 6, 32, false, false);
	RESNET_PLAN_NODE_FILL(1, 1, 7, 32, false, false);
	RESNET_PLAN_NODE_FILL(1, 2, 8, 18, false, false);
	RESNET_PLAN_NODE_FILL(1, 3, 9, 0, false, false);
	RESNET_PLAN_NODE_FILL(1, 4, 10, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(1, 5);

	RESNET_PLAN_NODE_FILL(2, 0, 12, 32, false, false);
	RESNET_PLAN_NODE_FILL(2, 1, 13, 32, false, false);
	RESNET_PLAN_NODE_FILL(2, 2, 14, 18, false, false);
	RESNET_PLAN_NODE_FILL(2, 3, 15, 0, false, false);
	RESNET_PLAN_NODE_FILL(2, 4, 16, 0, true, true);
	RESNET_PLAN_NODE_FILL(2, 5, 17, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(2, 6);

	RESNET_PLAN_NODE_FILL(3, 0, 19, 32, false, false);
	RESNET_PLAN_NODE_FILL(3, 1, 20, 32, false, false);
	RESNET_PLAN_NODE_FILL(3, 2, 21, 18, false, false);
	RESNET_PLAN_NODE_FILL(3, 3, 22, 0, false, false);
	RESNET_PLAN_NODE_FILL(3, 4, 23, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(3, 5);

	RESNET_PLAN_NODE_FILL(4, 0, 25, 32, false, false);
	RESNET_PLAN_NODE_FILL(4, 1, 26, 32, false, false);
	RESNET_PLAN_NODE_FILL(4, 2, 27, 18, false, false);
	RESNET_PLAN_NODE_FILL(4, 3, 28, 0, false, false);
	RESNET_PLAN_NODE_FILL(4, 4, 29, 0, true, true);
	RESNET_PLAN_NODE_FILL(4, 5, 30, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(4, 6);

	RESNET_PLAN_NODE_FILL(5, 0, 32, 32, false, false);
	RESNET_PLAN_NODE_FILL(5, 1, 33, 32, false, false);
	RESNET_PLAN_NODE_FILL(5, 2, 34, 18, false, false);
	RESNET_PLAN_NODE_FILL(5, 3, 35, 0, false, false);
	RESNET_PLAN_NODE_FILL(5, 4, 36, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(5, 5);

	RESNET_PLAN_NODE_FILL(6, 0, 38, 32, false, false);
	RESNET_PLAN_NODE_FILL(6, 1, 39, 32, false, false);
	RESNET_PLAN_NODE_FILL(6, 2, 40, 18, false, false);
	RESNET_PLAN_NODE_FILL(6, 3, 41, 0, false, false);
	RESNET_PLAN_NODE_FILL(6, 4, 42, 0, true, true);
	RESNET_PLAN_NODE_FILL(6, 5, 43, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(6, 6);

	RESNET_PLAN_NODE_FILL(7, 0, 45, 32, false, false);
	RESNET_PLAN_NODE_FILL(7, 1, 46, 32, false, false);
	RESNET_PLAN_NODE_FILL(7, 2, 47, 18, false, false);
	RESNET_PLAN_NODE_FILL(7, 3, 48, 0, false, false);
	RESNET_PLAN_NODE_FILL(7, 4, 49, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(7, 5);

	RESNET_PLAN_NODE_FILL(8, 0, 51, 32, false, false);
	RESNET_PLAN_NODE_FILL(8, 1, 52, 32, false, false);
	RESNET_PLAN_NODE_FILL(8, 2, 53, 18, false, false);
	RESNET_PLAN_NODE_FILL(8, 3, 54, 0, false, false);
	RESNET_PLAN_NODE_FILL(8, 4, 55, 0, true, true);
	RESNET_PLAN_NODE_FILL(8, 5, 56, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(8, 6);

	RESNET_PLAN_NODE_FILL(9, 0, 58, 32, false, false);
	RESNET_PLAN_NODE_FILL(9, 1, 59, 32, false, false);
	RESNET_PLAN_NODE_FILL(9, 2, 60, 18, false, false);
	RESNET_PLAN_NODE_FILL(9, 3, 61, 0, false, false);
	RESNET_PLAN_NODE_FILL(9, 4, 62, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(9, 5);

	RESNET_PLAN_NODE_FILL(10, 0, 64, 32, false, false);
	RESNET_PLAN_NODE_FILL(10, 1, 65, 32, false, false);
	RESNET_PLAN_NODE_FILL(10, 2, 66, 18, false, false);
	RESNET_PLAN_NODE_FILL(10, 3, 67, 0, false, false);
	RESNET_PLAN_NODE_FILL(10, 4, 68, 0, true, true);
	RESNET_PLAN_NODE_FILL(10, 5, 69, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(10, 6);

	RESNET_PLAN_NODE_FILL(11, 0, 71, 32, false, false);
	RESNET_PLAN_NODE_FILL(11, 1, 72, 32, false, false);
	RESNET_PLAN_NODE_FILL(11, 2, 73, 18, false, false);
	RESNET_PLAN_NODE_FILL(11, 3, 74, 0, false, false);
	RESNET_PLAN_NODE_FILL(11, 4, 75, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(11, 5);

	RESNET_PLAN_NODE_FILL(12, 0, 77, 32, false, false);
	RESNET_PLAN_NODE_FILL(12, 1, 78, 32, false, false);
	RESNET_PLAN_NODE_FILL(12, 2, 79, 18, false, false);
	RESNET_PLAN_NODE_FILL(12, 3, 80, 0, false, false);
	RESNET_PLAN_NODE_FILL(12, 4, 81, 0, true, true);
	RESNET_PLAN_NODE_FILL(12, 5, 82, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(12, 6);

	RESNET_PLAN_NODE_FILL(13, 0, 84, 32, false, false);
	RESNET_PLAN_NODE_FILL(13, 1, 85, 32, false, false);
	RESNET_PLAN_NODE_FILL(13, 2, 86, 18, false, false);
	RESNET_PLAN_NODE_FILL(13, 3, 87, 0, false, false);
	RESNET_PLAN_NODE_FILL(13, 4, 88, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(13, 5);

	RESNET_PLAN_NODE_FILL(14, 0, 90, 32, false, false);
	RESNET_PLAN_NODE_FILL(14, 1, 91, 32, false, false);
	RESNET_PLAN_NODE_FILL(14, 2, 92, 18, false, false);
	RESNET_PLAN_NODE_FILL(14, 3, 93, 0, false, false);
	RESNET_PLAN_NODE_FILL(14, 4, 94, 0, true, true);
	RESNET_PLAN_NODE_FILL(14, 5, 95, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(14, 6);

	RESNET_PLAN_NODE_FILL(15, 0, 97, 32, false, false);
	RESNET_PLAN_NODE_FILL(15, 1, 98, 32, false, false);
	RESNET_PLAN_NODE_FILL(15, 2, 99, 18, false, false);
	RESNET_PLAN_NODE_FILL(15, 3, 100, 0, false, false);
	RESNET_PLAN_NODE_FILL(15, 4, 101, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(15, 5);

	RESNET_PLAN_NODE_FILL(16, 0, 103, 32, false, false);
	RESNET_PLAN_NODE_FILL(16, 1, 104, 32, false, false);
	RESNET_PLAN_NODE_FILL(16, 2, 105, 18, false, false);
	RESNET_PLAN_NODE_FILL(16, 3, 106, 0, false, false);
	RESNET_PLAN_NODE_FILL(16, 4, 111, 0, true, true);
	RESNET_PLAN_NODE_FILL(16, 5, 112, 32, false, false);
	//RESNET_PLAN_NODE_FILL(16, 3, 106, 0, false, false);
	//RESNET_PLAN_NODE_FILL(16, 4, 107, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(16, 6);

        // plan17 needs to be called before plan16
	RESNET_PLAN_NODE_FILL(17, 0, 109, 32, false, false);
	RESNET_PLAN_NODE_FILL(17, 1, 110, 32, false, true);

	RESNET_PLAN_SET_NODE_NUM(17, 2); // no output

	RESNET_PLAN_NODE_FILL(18, 0, 114, 32, false, false);
	RESNET_PLAN_NODE_FILL(18, 1, 115, 32, false, false);
	RESNET_PLAN_NODE_FILL(18, 2, 116, 18, false, false);
	RESNET_PLAN_NODE_FILL(18, 3, 117, 0, false, false);
	RESNET_PLAN_NODE_FILL(18, 4, 118, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(18, 5);

	RESNET_PLAN_NODE_FILL(19, 0, 120, 32, false, false);
	RESNET_PLAN_NODE_FILL(19, 1, 121, 32, false, false);
	RESNET_PLAN_NODE_FILL(19, 2, 122, 18, false, false);
	RESNET_PLAN_NODE_FILL(19, 3, 123, 0, false, false);
	RESNET_PLAN_NODE_FILL(19, 4, 124, 0, true, true);
	RESNET_PLAN_NODE_FILL(19, 5, 125, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(19, 6);

	RESNET_PLAN_NODE_FILL(20, 0, 127, 32, false, false);
	RESNET_PLAN_NODE_FILL(20, 1, 128, 32, false, false);
	RESNET_PLAN_NODE_FILL(20, 2, 129, 18, false, false);
	RESNET_PLAN_NODE_FILL(20, 3, 130, 0, false, false);
	RESNET_PLAN_NODE_FILL(20, 4, 131, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(20, 5);

	RESNET_PLAN_NODE_FILL(21, 0, 133, 32, false, false);
	RESNET_PLAN_NODE_FILL(21, 1, 134, 32, false, false);
	RESNET_PLAN_NODE_FILL(21, 2, 135, 18, false, false);
	RESNET_PLAN_NODE_FILL(21, 3, 136, 0, false, false);
	RESNET_PLAN_NODE_FILL(21, 4, 137, 0, true, true);
	RESNET_PLAN_NODE_FILL(21, 5, 138, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(21, 6);

	RESNET_PLAN_NODE_FILL(22, 0, 140, 32, false, false);
	RESNET_PLAN_NODE_FILL(22, 1, 141, 32, false, false);
	RESNET_PLAN_NODE_FILL(22, 2, 142, 18, false, false);
	RESNET_PLAN_NODE_FILL(22, 3, 143, 0, false, false);
	RESNET_PLAN_NODE_FILL(22, 4, 144, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(22, 5);

	RESNET_PLAN_NODE_FILL(23, 0, 146, 32, false, false);
	RESNET_PLAN_NODE_FILL(23, 1, 147, 32, false, false);
	RESNET_PLAN_NODE_FILL(23, 2, 148, 18, false, false);
	RESNET_PLAN_NODE_FILL(23, 3, 149, 0, false, false);
	RESNET_PLAN_NODE_FILL(23, 4, 150, 0, true, true);
	RESNET_PLAN_NODE_FILL(23, 5, 151, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(23, 6);

	RESNET_PLAN_NODE_FILL(24, 0, 153, 32, false, false);
	RESNET_PLAN_NODE_FILL(24, 1, 154, 32, false, false);
	RESNET_PLAN_NODE_FILL(24, 2, 155, 18, false, false);
	RESNET_PLAN_NODE_FILL(24, 3, 156, 0, false, false);
	RESNET_PLAN_NODE_FILL(24, 4, 157, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(24, 5);

	RESNET_PLAN_NODE_FILL(25, 0, 159, 32, false, false);
	RESNET_PLAN_NODE_FILL(25, 1, 160, 32, false, false);
	RESNET_PLAN_NODE_FILL(25, 2, 161, 18, false, false);
	RESNET_PLAN_NODE_FILL(25, 3, 162, 0, false, false);
	RESNET_PLAN_NODE_FILL(25, 4, 163, 0, true, true);
	RESNET_PLAN_NODE_FILL(25, 5, 164, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(25, 6);

	RESNET_PLAN_NODE_FILL(26, 0, 166, 32, false, false);
	RESNET_PLAN_NODE_FILL(26, 1, 167, 32, false, false);
	RESNET_PLAN_NODE_FILL(26, 2, 168, 18, false, false);
	RESNET_PLAN_NODE_FILL(26, 3, 169, 0, false, false);
	RESNET_PLAN_NODE_FILL(26, 4, 170, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(26, 5);

	RESNET_PLAN_NODE_FILL(27, 0, 172, 32, false, false);
	RESNET_PLAN_NODE_FILL(27, 1, 173, 32, false, false);
	RESNET_PLAN_NODE_FILL(27, 2, 174, 18, false, false);
	RESNET_PLAN_NODE_FILL(27, 3, 175, 0, false, false);
	RESNET_PLAN_NODE_FILL(27, 4, 176, 0, true, true);
	RESNET_PLAN_NODE_FILL(27, 5, 177, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(27, 6);

	RESNET_PLAN_NODE_FILL(28, 0, 179, 32, false, false);
	RESNET_PLAN_NODE_FILL(28, 1, 180, 32, false, false);
	RESNET_PLAN_NODE_FILL(28, 2, 181, 18, false, false);
	RESNET_PLAN_NODE_FILL(28, 3, 182, 0, false, false);
	RESNET_PLAN_NODE_FILL(28, 4, 183, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(28, 5);

	RESNET_PLAN_NODE_FILL(29, 0, 185, 32, false, false);
	RESNET_PLAN_NODE_FILL(29, 1, 186, 32, false, false);
	RESNET_PLAN_NODE_FILL(29, 2, 187, 18, false, false);
	RESNET_PLAN_NODE_FILL(29, 3, 188, 0, false, false);
	RESNET_PLAN_NODE_FILL(29, 4, 189, 0, true, true);
	RESNET_PLAN_NODE_FILL(29, 5, 190, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(29, 6);

	RESNET_PLAN_NODE_FILL(30, 0, 192, 32, false, false);
	RESNET_PLAN_NODE_FILL(30, 1, 193, 32, false, false);
	RESNET_PLAN_NODE_FILL(30, 2, 194, 18, false, false);
	RESNET_PLAN_NODE_FILL(30, 3, 195, 0, false, false);
	RESNET_PLAN_NODE_FILL(30, 4, 196, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(30, 5);

	RESNET_PLAN_NODE_FILL(31, 0, 198, 32, false, false);
	RESNET_PLAN_NODE_FILL(31, 1, 199, 32, false, false);
	RESNET_PLAN_NODE_FILL(31, 2, 200, 18, false, false);
	RESNET_PLAN_NODE_FILL(31, 3, 201, 0, false, false);
	//RESNET_PLAN_NODE_FILL(31, 4, 202, 32, false, false);
	RESNET_PLAN_NODE_FILL(32, 4, 206, 0, true, true);
	RESNET_PLAN_NODE_FILL(32, 5, 207, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(31, 6);

        // plan 32 needs to execute before plan 31
	RESNET_PLAN_NODE_FILL(32, 0, 204, 32, false, false);
	RESNET_PLAN_NODE_FILL(32, 1, 205, 32, false, true);

	RESNET_PLAN_SET_NODE_NUM(32, 2);

	RESNET_PLAN_NODE_FILL(33, 0, 209, 32, false, false);
	RESNET_PLAN_NODE_FILL(33, 1, 210, 32, false, false);
	RESNET_PLAN_NODE_FILL(33, 2, 211, 18, false, false);
	RESNET_PLAN_NODE_FILL(33, 3, 212, 0, false, false);
	RESNET_PLAN_NODE_FILL(33, 4, 213, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(33, 5);

	RESNET_PLAN_NODE_FILL(34, 0, 215, 32, false, false);
	RESNET_PLAN_NODE_FILL(34, 1, 216, 32, false, false);
	RESNET_PLAN_NODE_FILL(34, 2, 217, 18, false, false);
	RESNET_PLAN_NODE_FILL(34, 3, 218, 0, false, false);
	RESNET_PLAN_NODE_FILL(34, 4, 219, 0, true, true);
	RESNET_PLAN_NODE_FILL(34, 5, 220, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(34, 6);

	RESNET_PLAN_NODE_FILL(35, 0, 222, 32, false, false);
	RESNET_PLAN_NODE_FILL(35, 1, 223, 32, false, false);
	RESNET_PLAN_NODE_FILL(35, 2, 224, 18, false, false);
	RESNET_PLAN_NODE_FILL(35, 3, 225, 0, false, false);
	RESNET_PLAN_NODE_FILL(35, 4, 226, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(35, 5);

	RESNET_PLAN_NODE_FILL(36, 0, 228, 32, false, false);
	RESNET_PLAN_NODE_FILL(36, 1, 229, 32, false, false);
	RESNET_PLAN_NODE_FILL(36, 2, 230, 18, false, false);
	RESNET_PLAN_NODE_FILL(36, 3, 231, 0, false, false);
	RESNET_PLAN_NODE_FILL(36, 4, 232, 0, true, true);
	RESNET_PLAN_NODE_FILL(36, 5, 233, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(36, 6);

	RESNET_PLAN_NODE_FILL(37, 0, 235, 32, false, false);
	RESNET_PLAN_NODE_FILL(37, 1, 236, 32, false, false);
	RESNET_PLAN_NODE_FILL(37, 2, 237, 18, false, false);
	RESNET_PLAN_NODE_FILL(37, 3, 238, 0, false, false);
	RESNET_PLAN_NODE_FILL(37, 4, 239, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(37, 5);

	RESNET_PLAN_NODE_FILL(38, 0, 241, 32, false, false);
	RESNET_PLAN_NODE_FILL(38, 1, 242, 32, false, false);
	RESNET_PLAN_NODE_FILL(38, 2, 243, 18, false, false);
	RESNET_PLAN_NODE_FILL(38, 3, 244, 0, false, false);
	RESNET_PLAN_NODE_FILL(38, 4, 245, 0, true, true);
	RESNET_PLAN_NODE_FILL(38, 5, 246, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(38, 6);

	RESNET_PLAN_NODE_FILL(39, 0, 248, 32, false, false);
	RESNET_PLAN_NODE_FILL(39, 1, 249, 32, false, false);
	RESNET_PLAN_NODE_FILL(39, 2, 250, 18, false, false);
	RESNET_PLAN_NODE_FILL(39, 3, 251, 0, false, false);
	RESNET_PLAN_NODE_FILL(39, 4, 252, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(39, 5);

	RESNET_PLAN_NODE_FILL(40, 0, 254, 32, false, false);
	RESNET_PLAN_NODE_FILL(40, 1, 255, 32, false, false);
	RESNET_PLAN_NODE_FILL(40, 2, 256, 18, false, false);
	RESNET_PLAN_NODE_FILL(40, 3, 257, 0, false, false);
	RESNET_PLAN_NODE_FILL(40, 4, 258, 0, true, true);
	RESNET_PLAN_NODE_FILL(40, 5, 259, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(40, 6);

	RESNET_PLAN_NODE_FILL(41, 0, 261, 32, false, false);
	RESNET_PLAN_NODE_FILL(41, 1, 262, 32, false, false);
	RESNET_PLAN_NODE_FILL(41, 2, 263, 18, false, false);
	RESNET_PLAN_NODE_FILL(41, 3, 264, 0, false, false);
	RESNET_PLAN_NODE_FILL(41, 4, 265, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(41, 5);

	RESNET_PLAN_NODE_FILL(42, 0, 267, 32, false, false);
	RESNET_PLAN_NODE_FILL(42, 1, 268, 32, false, false);
	RESNET_PLAN_NODE_FILL(42, 2, 269, 18, false, false);
	RESNET_PLAN_NODE_FILL(42, 3, 270, 0, false, false);
	RESNET_PLAN_NODE_FILL(42, 4, 271, 0, true, true);
	RESNET_PLAN_NODE_FILL(42, 5, 272, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(42, 6);

	RESNET_PLAN_NODE_FILL(43, 0, 274, 32, false, false);
	RESNET_PLAN_NODE_FILL(43, 1, 275, 32, false, false);
	RESNET_PLAN_NODE_FILL(43, 2, 276, 18, false, false);
	RESNET_PLAN_NODE_FILL(43, 3, 277, 0, false, false);
	RESNET_PLAN_NODE_FILL(43, 4, 278, 32, false, false);

	RESNET_PLAN_SET_NODE_NUM(43, 5);

	RESNET_PLAN_NODE_FILL(44, 0, 280, 32, false, false);
	RESNET_PLAN_NODE_FILL(44, 1, 281, 32, false, false);
	RESNET_PLAN_NODE_FILL(44, 2, 282, 18, false, false);
	RESNET_PLAN_NODE_FILL(44, 3, 283, 0, false, false);
	RESNET_PLAN_NODE_FILL(44, 4, 284, 0, true, false);
	RESNET_PLAN_NODE_FILL(44, 5, 285, 1, false, false);
	//RESNET_PLAN_NODE_FILL(44, 6, 286, 22, false, false);
	RESNET_PLAN_NODE_FILL(44, 6, 287, 9, false, false);
	RESNET_PLAN_NODE_FILL(44, 7, 288, 25, false, false);

	RESNET_PLAN_SET_NODE_NUM(44, 8);
}

void __assert_fail(const char * assertion, const char * file, unsigned int line, const char * function);
void __assert_fail(const char * assertion, const char * file, unsigned int line, const char * function) {
    DMSG("ASSERT FAIL: %s,line:%d, func:%s, file:%s\n", assertion, line, function, file);
    abort();
}

static unsigned get_max_op_id(); 
static unsigned get_max_op_id() { /* op_id start from 0 */
    return ns(Operator_vec_len(model_operators)) - 1;
}

static unsigned get_activation_type(unsigned cur_id); 
static unsigned get_activation_type(unsigned cur_id) {
    ns(AddOptions_table_t) add_option = (ns(AddOptions_table_t)) ns(Operator_builtin_options(ns(Operator_vec_at(model_operators, cur_id))));
    enum ActivationFunctionType type = ns(AddOptions_fused_activation_function(add_option));
    return (unsigned)type;
}

static unsigned count_output_tensor_use_frequency(unsigned cur_id);
static unsigned count_output_tensor_use_frequency(unsigned cur_id) {
    unsigned tid, op_id, in, inp_tid, freq;
    unsigned MAX_OP_ID = get_max_op_id(); 
    flatbuffers_int32_vec_t inputs, outputs;

    outputs = ns(Operator_outputs(ns(Operator_vec_at(model_operators, cur_id))));
    tid = flatbuffers_int32_vec_at(outputs, 0);

    freq = 0;
    for (op_id = cur_id; op_id < MAX_OP_ID; op_id++) {
        inputs = ns(Operator_inputs(ns(Operator_vec_at(model_operators, op_id))));
        in = flatbuffers_int32_vec_len(inputs);
        for (int j = 0; j < in; j ++) {
            inp_tid = flatbuffers_int32_vec_at(inputs, j);
            if (inp_tid == tid)
                freq += 1;
        }
    }
    return freq;
}

static bool is_second_input_variable(unsigned cur_id); 
static bool is_second_input_variable(unsigned cur_id) {
    ns(Buffer_table_t) buffer;
    flatbuffers_uint8_vec_t data;
    flatbuffers_int32_vec_t inputs;
    unsigned tid, buffer_id;

    // get second input buf
    inputs = ns(Operator_inputs(ns(Operator_vec_at(model_operators, cur_id))));
    tid = flatbuffers_int32_vec_at(inputs, 1);
    buffer_id = ns(Tensor_buffer(ns(Tensor_vec_at(model_tensors, tid))));
    buffer = ns(Buffer_vec_at(model_buffers, buffer_id));
    data = ns(Buffer_data(buffer));

    return data == NULL;
}

static void * get_tensor_buffer_data(int tensor_id, bool is_float); 
static void * get_tensor_buffer_data(int tensor_id, bool is_float) {
    ns(Buffer_table_t) buffer;
    flatbuffers_uint8_vec_t data;
    unsigned buffer_id;
    int len;
    float *p;
    unsigned *u;

    buffer_id = ns(Tensor_buffer(ns(Tensor_vec_at(model_tensors, tensor_id))));
    buffer = ns(Buffer_vec_at(model_buffers, buffer_id));
    data = ns(Buffer_data(buffer));
    len = flatbuffers_uint8_vec_len(data);

    if (data == NULL) { 
        DMSG("%s :data is NULL!\n", __func__);
    } else {
        //DMSG("\nbuffer_id: %u, data pointer start:%p, end:%p, %4f\n", buffer_id, data, data+len, 0.1234);
        if (is_float) {
            p = (float *)data;
            //DMSG("data len:%d first 4 floats: %6f,%6f,%6f,%6f\n\n", len, p[0], p[1], p[2], p[3]);
        } else {
            u = (unsigned *)data;
            //DMSG("\ndata len:%d first 4 unsigned : %u %u %u %u\n\n", len, u[0], u[1], u[2], u[3]);
        }
    }

    // could be NULL
    return (void *)data;
}

static void get_tensor_info(ns(Tensor_vec_t) tensors, int id); 
static void get_tensor_info(ns(Tensor_vec_t) tensors, int id) {
    flatbuffers_int32_vec_t shape;
    int shape_len;
    DMSG("tensor id: %d tensor name: %s\n",id, ns(Tensor_name(ns(Tensor_vec_at(tensors, id)))));
    
    get_tensor_buffer_data(id, true);
    shape = ns(Tensor_shape(ns(Tensor_vec_at(tensors, id))));
    shape_len = flatbuffers_int32_vec_len(shape);
    DMSG("tensor shape: ");
    for (int j = 0; j < shape_len; j ++) {
        DMSG(" %d, ", flatbuffers_int32_vec_at(shape, j));
    }
    DMSG("\n\n");
}



static void list_all_buffers(ns(Buffer_vec_t) buffers); 
static void list_all_buffers(ns(Buffer_vec_t) buffers) {
    int num = ns(Buffer_vec_len(buffers));
    ns(Buffer_table_t) buffer;
    flatbuffers_uint8_vec_t data;
    int len;
    for (int i = 0; i < num; i++) {
        DMSG("\n buffer id: %d\n", i);
        buffer = ns(Buffer_vec_at(buffers, i));
        data = ns(Buffer_data(buffer));
        if (data != NULL) {
            len = flatbuffers_uint8_vec_len(data);
            DMSG("data len:%d first 4 bytes: %u %u %u %u\n", len,
                data[0], data[1],data[2],data[3]);
        } else {
            DMSG("data is NULL\n");
        }
    }
} 
static int list_all_operators(ns(Operator_vec_t) operators, 
                            ns(OperatorCode_vec_t) opcodes,
                            ns(Tensor_vec_t) tensors); 

static int list_all_operators(ns(Operator_vec_t) operators, 
                            ns(OperatorCode_vec_t) opcodes,
                            ns(Tensor_vec_t) tensors) {
    int num = ns(Operator_vec_len(operators));
    unsigned opcode_id;
    flatbuffers_int32_vec_t inputs, outputs;
    int in,out;

    if (num <=0)
        return -1;
    for (int i = 0; i < num; i++) {
        DMSG(" operator id: %d\n", i);

        opcode_id = ns(Operator_opcode_index(ns(Operator_vec_at(operators, i))));

        DMSG("opcode custom name: %s \n",ns(OperatorCode_custom_code(ns(OperatorCode_vec_at(opcodes, opcode_id)))));
        DMSG("builtin operator : %d \n",ns(OperatorCode_builtin_code(ns(OperatorCode_vec_at(opcodes, opcode_id)))));

        inputs = ns(Operator_inputs(ns(Operator_vec_at(operators, i))));
        outputs = ns(Operator_outputs(ns(Operator_vec_at(operators, i))));
        in = flatbuffers_int32_vec_len(inputs);
        out = flatbuffers_int32_vec_len(outputs);

        DMSG("\ninput tensors:\n");
        for (int j = 0; j < in; j ++) {
            get_tensor_info(tensors, flatbuffers_int32_vec_at(inputs, j));
        }

        DMSG("\noutput tensors:\n");
        for (int j = 0; j < out; j ++) {
            get_tensor_info(tensors, flatbuffers_int32_vec_at(outputs, j));
        }
        DMSG("\n\n");
    }
    return 0;
}

static int list_all_tensors(ns(Tensor_vec_t) tensors); 
static int list_all_tensors(ns(Tensor_vec_t) tensors) {
    int num = ns(Tensor_vec_len(tensors));
    if (num <=0)
        return -1;
    for (int i = 0; i < num; i++) {
        DMSG("tensor id: %d tensor name: %s\n",i, ns(Tensor_name(ns(Tensor_vec_at(tensors, i)))));
    }
    return 0;
}

// initialize model from model_buffer
int initialize_tflite_model_from_buffer(void *buffer)
{
    flatbuffers_string_t description;
    unsigned opcode_index;

    if (!(model = ns(Model_as_root(buffer)))) {
        DMSG("Monster not available\n");
        return -1;
    }

    if (ns(Model_version(model)) != 1) {
        DMSG("model version is %d\n", ns(Model_version(model)));
    }

    description = ns(Model_description(model));
    DMSG("description :%s\n", description);

    model_subgraphs = ns(Model_subgraphs(model));
    DMSG("subgraph name: %s\n", ns(SubGraph_name(ns(SubGraph_vec_at(model_subgraphs, 0)))));

    model_opcodes = ns(Model_operator_codes(model));
    model_sub0 = ns(SubGraph_vec_at(model_subgraphs, 0));
    model_operators = ns(SubGraph_operators(model_sub0));
    model_tensors = ns(SubGraph_tensors(model_sub0));
    model_buffers = ns(Model_buffers(model));

    DMSG("Flatbuffer model initialized!\n");
    //DMSG("list all operators\n");
    //list_all_operators(model_operators, model_opcodes, model_tensors);

    //list_all_buffers(model_buffers);

    return 0;
}


// get operator code op_code from operator id op_id
static unsigned get_op_code(unsigned op_id); 
static unsigned get_op_code(unsigned op_id) {
    unsigned opcode_id, op_code;
    opcode_id = ns(Operator_opcode_index(ns(Operator_vec_at(model_operators, op_id))));
    op_code = ns(OperatorCode_builtin_code(ns(OperatorCode_vec_at(model_opcodes, opcode_id))));

    return op_code;
}


// If we load mobilenet tflite model into TEE, it will take 61MB
// We do not have extra memory for output buf, so we want to reuse
// the space occupied by linear layer's weights which won't be
// used by TEE, and they take large chunk of memory
// Example 1: 
// layer name           conv_pw_13 (Conv2D) 
// output shape         (None, 7, 7, 1228)
// #parameters:         1257472 
// operator ID:         173
// weights index:       1
// tensor ID:           278
// Example 2:
// layer name           obf_conv_preds (Conv2D)
// output shape         (None, 1, 1, 1200)
// #parameters:         1228800
// operator ID:         181
// weights index:       1
// tensor ID:           280
// example tensors: 274, 278, 281

void init_output_buf() {
#ifdef USE_MOBILENETS
    outbuf1 = (float *)get_tensor_buffer_data(278, true); 
    outbuf2 = (float *)get_tensor_buffer_data(281, true); 
    tempbuf = (float *)get_tensor_buffer_data(274, true); 
#else
    outbuf1 = (float *)TEE_Malloc(SN_MAX_BUF_SIZE, TEE_MALLOC_FILL_ZERO);
    outbuf2 = (float *)TEE_Malloc(SN_MAX_BUF_SIZE, TEE_MALLOC_FILL_ZERO);
    tempbuf = (float *)TEE_Malloc(SN_MAX_BUF_SIZE, TEE_MALLOC_FILL_ZERO);
#endif
    if (outbuf1 == NULL || outbuf2 == NULL || tempbuf == NULL)
        DMSG("ERROR! allocating buffer for outbuf1||outbuf2 failed!\n");
    else
        DMSG("outbuf1:%p, outbuf2:%p, tempbuf:%p \n", outbuf1, outbuf2, tempbuf);

}


// get the shape dimention at index dim_idx from the input tensor input_t_idx of operator op_id
// input_idx for operator: inputs[ t1, t2, t3 ... ]
// dim_idx for tensor: shape [ d1, d2,...]
static unsigned get_operator_tensor_shape_dim(unsigned op_id, unsigned input_idx, unsigned dim_idx); 
static unsigned get_operator_tensor_shape_dim(unsigned op_id, unsigned input_idx, unsigned dim_idx) {
    flatbuffers_int32_vec_t inputs;
    flatbuffers_int32_vec_t shape;
    unsigned dim;
    unsigned input_tensor_id, inputs_len, shape_len;

    inputs = ns(Operator_inputs(ns(Operator_vec_at(model_operators, op_id))));

    // check input_idx bound
    inputs_len = flatbuffers_int32_vec_len(inputs);
    assert(input_idx < inputs_len);

    input_tensor_id = flatbuffers_int32_vec_at(inputs, input_idx);

    // input tensors shape (include input and weights)
    //  for input (batch, h, w, c)
    //  for weights shape is uncertain
    shape = ns(Tensor_shape(ns(Tensor_vec_at(model_tensors, input_tensor_id))));

    // check dim_idx bound
    shape_len = flatbuffers_int32_vec_len(shape);
    assert(dim_idx < shape_len);

    dim = flatbuffers_int32_vec_at(shape, dim_idx);
    return dim;
}

// get the input tensor input_idx's data buffer of operator op_id
static void *get_operator_tensor_buffer(unsigned op_id, unsigned input_idx, bool is_float); 
static void *get_operator_tensor_buffer(unsigned op_id, unsigned input_idx, bool is_float) {
    flatbuffers_int32_vec_t inputs;
    unsigned inputs_len, input_tensor_id;
    void *data;

    inputs = ns(Operator_inputs(ns(Operator_vec_at(model_operators, op_id))));

    // check input_idx bound
    inputs_len = flatbuffers_int32_vec_len(inputs);
    assert(input_idx < inputs_len);

    input_tensor_id = flatbuffers_int32_vec_at(inputs, input_idx);
    data = (void *)get_tensor_buffer_data(input_tensor_id, is_float); 

    return data;
}

static void shadownet_linear_transform(unsigned cur_id, float *input, float *output); 
static void shadownet_linear_transform(unsigned cur_id, float *input, float *output) {
    int H, W, M, N;
    int h, w, n;
    int idx_from, idx_rand;
    float scalar;
    int *lt_obfweights;
    float *lt_rbias;

    // input tensor shape (batch, h, w, c)
    H = get_operator_tensor_shape_dim(cur_id, LT_INPUT_IDX, HEIGHT_IDX);
    W = get_operator_tensor_shape_dim(cur_id, LT_INPUT_IDX, WIDTH_IDX);
    M = get_operator_tensor_shape_dim(cur_id, LT_INPUT_IDX, CHN_IDX);

    // weight tensor shape (2, units)
    N = get_operator_tensor_shape_dim(cur_id, LT_WEIGHT1_IDX, UNIT_IDX);

    DMSG("%s input:%p, output:%p H:%d, W:%d, M:%d, N:%d\n", __func__, input, output, H, W, M, N);

    // weights buffer 
    lt_obfweights = (int *)get_operator_tensor_buffer(cur_id, LT_WEIGHT1_IDX, false); 
    lt_rbias = (float *)get_operator_tensor_buffer(cur_id, LT_WEIGHT2_IDX, true); 

    assert(lt_obfweights != NULL);
    assert(lt_rbias != NULL);

    DMSG("%s obfweights:%p, rbias:%p\n", __func__, lt_obfweights, lt_rbias);
    for (h = 0; h < H; ++h) {
        for (w = 0; w < W; ++w) {
            for (n = 0; n < N; ++n) {
                idx_from = lt_obfweights[n]; 
                idx_rand = lt_obfweights[N + n]; 
                scalar = lt_rbias[n]; 

                output[(h * W * N) + (w * N) + n] = input[(h * W * M) + (w * M) + idx_from] * scalar +  input[(h * W * M) + (w * M) + idx_rand]; 
            }
        }
    }
    DMSG("%s finish!", __func__);
}

static void shadownet_mul_add(unsigned cur_id, float *input, float *output, bool use_buf, bool store_buf); 
static void shadownet_mul_add(unsigned cur_id, float *input, float *output, bool use_buf, bool store_buf) {

    int H, W, C;
    int i,j;
    float *mul_scalars, *bias;
    unsigned type = get_activation_type(cur_id+1); 

    // input tensor shape (batch, h, w, c)
    H = get_operator_tensor_shape_dim(cur_id, 0/*INPUT_IDX*/, 1/*HEIGHT_IDX*/);
    W = get_operator_tensor_shape_dim(cur_id, 0/*INPUT_IDX*/, 2/*WIDTH_IDX*/);
    C = get_operator_tensor_shape_dim(cur_id, 0/*INPUT_IDX*/, 3/*CHANNEL_IDX*/);

    mul_scalars = (float *)get_operator_tensor_buffer(cur_id, 1/*WEIGHT_IDX*/, true /*is_float*/); 
    // ADD biases
    bias = (float *)get_operator_tensor_buffer(cur_id+1, 1/*WEIGHT_IDX*/, true /*is_float*/); 

    DMSG("%s H:%d, W:%d, C:%d, weights:%p\n", __func__, H, W, C, mul_scalars);
    assert(mul_scalars != NULL);

    for(j = 0; j < H*W; ++j){
        neon_muladd(input + j*C, mul_scalars, bias, input + j*C, C);
    }

    // TODO neon_relu not implemented yet, use relu6
    if (type == ACT_RELU6 || type == ACT_RELU)
        neon_relu6(input, input, H*W*C); 

    if (store_buf) { // write to tempbuf
        for(j = 0; j < H*W; ++j)
            for(i = 0; i < C; ++i)
                tempbuf[j*C + i] = input[j*C + i];
    }

}

static void shadownet_mul(unsigned cur_id, float *input, float *output); 
static void shadownet_mul(unsigned cur_id, float *input, float *output) {
    int H, W, C;
    int i,j;
    float *mul_scalars;

    // input tensor shape (batch, h, w, c)
    H = get_operator_tensor_shape_dim(cur_id, 0/*INPUT_IDX*/, 1/*HEIGHT_IDX*/);
    W = get_operator_tensor_shape_dim(cur_id, 0/*INPUT_IDX*/, 2/*WIDTH_IDX*/);
    C = get_operator_tensor_shape_dim(cur_id, 0/*INPUT_IDX*/, 3/*CHANNEL_IDX*/);

    mul_scalars = (float *)get_operator_tensor_buffer(cur_id, 1/*WEIGHT_IDX*/, true /*is_float*/); 

    DMSG("%s H:%d, W:%d, C:%d, weights:%p\n", __func__, H, W, C, mul_scalars);
    assert(mul_scalars != NULL);

    for(j = 0; j < H*W; ++j){
        for(i = 0; i < C; ++i){
            output[j*C + i] = input[j*C+i] * mul_scalars[i];
        }
    }

}

static void shadownet_add(unsigned cur_id, float *input, float *output, bool use_buf, bool store_buf); 
static void shadownet_add(unsigned cur_id, float *input, float *output, bool use_buf, bool store_buf) {
    int H, W, C;
    int i,j;
    float *second_input;
    //float *add_biases;
    unsigned type = get_activation_type(cur_id); 

    // input tensor shape (batch, h, w, c)
    H = get_operator_tensor_shape_dim(cur_id, 0/*INPUT_IDX*/, 1/*HEIGHT_IDX*/);
    W = get_operator_tensor_shape_dim(cur_id, 0/*INPUT_IDX*/, 2/*WIDTH_IDX*/);
    C = get_operator_tensor_shape_dim(cur_id, 0/*INPUT_IDX*/, 3/*CHANNEL_IDX*/);

    if (use_buf) {// use buf is true, second_input from temptbuf
        second_input = tempbuf;
    } else {
        second_input = (float *)get_operator_tensor_buffer(cur_id, 1/*WEIGHT_IDX*/, true /*is_float*/); 
    }

    DMSG("%s H:%d, W:%d, C:%d, second_input:%p\n", __func__, H, W, C, second_input);
    //assert(add_biases!= NULL);

    if (use_buf) {
        if (type == ACT_RELU6) {
            for(j = 0; j < H*W; ++j)
                for(i = 0; i < C; ++i)
                    output[j*C + i] = relu6_activate(input[j*C + i] + second_input[j*C + i]);
        } else if (type == ACT_RELU) {
            for(j = 0; j < H*W; ++j)
                for(i = 0; i < C; ++i)
                    output[j*C + i] = relu_activate(input[j*C + i] + second_input[j*C + i]);
        } else if (type == ACT_NONE) {
            for(j = 0; j < H*W; ++j)
                for(i = 0; i < C; ++i)
                    output[j*C + i] = input[j*C + i] + second_input[j*C + i];
        }
    } else {
        if (type == ACT_RELU6) {
            for(j = 0; j < H*W; ++j)
                for(i = 0; i < C; ++i)
                    output[j*C + i] = relu6_activate(input[j*C + i] + second_input[i]);
        } else if (type == ACT_RELU) {
            for(j = 0; j < H*W; ++j)
                for(i = 0; i < C; ++i)
                    output[j*C + i] = relu_activate(input[j*C + i] + second_input[i]);
        } else if (type == ACT_NONE) {
            for(j = 0; j < H*W; ++j)
                for(i = 0; i < C; ++i)
                    output[j*C + i] = input[j*C + i] + second_input[i];
        }
    }

    if (store_buf) { // write to tempbuf
        for(j = 0; j < H*W; ++j)
            for(i = 0; i < C; ++i)
                tempbuf[j*C + i] = output[j*C + i];
    }
}

static void shadownet_add_mask(unsigned cur_id, float *input, float *output); 
static void shadownet_add_mask(unsigned cur_id, float *input, float *output) {
    int H, W, C;
    int i;
    float *add_masks;

    // input tensor shape (batch, h, w, c)
    H = get_operator_tensor_shape_dim(cur_id, 0/*INPUT_IDX*/, 1/*HEIGHT_IDX*/);
    W = get_operator_tensor_shape_dim(cur_id, 0/*INPUT_IDX*/, 2/*WIDTH_IDX*/);
    C = get_operator_tensor_shape_dim(cur_id, 0/*INPUT_IDX*/, 3/*CHANNEL_IDX*/);

    add_masks = (float *)get_operator_tensor_buffer(cur_id, 1/*WEIGHT_IDX*/, true /*is_float*/); 

    DMSG("%s H:%d, W:%d, C:%d, weights:%p\n", __func__, H, W, C, add_masks);
    assert(add_masks!= NULL);

#ifdef OPTIMIZE_MUL_ADD
    neon_muladd_fixed_scalar(add_masks, 1.0, input, output, H*W*C); 
#else
    for(i = 0; i < H*W*C; ++i){
        output[i] = input[i] + add_masks[i];
    }
#endif
}

static void softmax(float *input, int n, float temp, float *output);
static void softmax(float *input, int n, float temp, float *output)
{
    int i;
    float e;
    float sum = 0;
    float largest = -FLT_MAX;
    for(i = 0; i < n; ++i){
        if(input[i] > largest) largest = input[i];
    }
    for(i = 0; i < n; ++i){
        e = ta_exp(input[i]/temp - largest/temp);
        sum += e;
        output[i] = e;
    }
    for(i = 0; i < n; ++i){
        output[i] /= sum;
    }
    return;
}

static float get_softmax_beta(unsigned op_id); 
static float get_softmax_beta(unsigned op_id) {
    assert(op_id < ns(Operator_vec_len(model_operators)));
    assert(SOFTMAX == get_op_code(op_id));
    ns(SoftmaxOptions_table_t) softmax_option = (ns(SoftmaxOptions_table_t)) ns(Operator_builtin_options(ns(Operator_vec_at(model_operators, op_id))));
    float beta = ns(SoftmaxOptions_beta(softmax_option));
    DMSG("softmax option beta: %6f\n", beta);
    return beta;
}

static void shadownet_softmax(unsigned cur_id, float *input, float *output);
static void shadownet_softmax(unsigned cur_id, float *input, float *output)
{
    int C;
    float beta;

    beta = get_softmax_beta(cur_id);
    // input tensor shape (batch, c)
    C = get_operator_tensor_shape_dim(cur_id, 0/*INPUT_IDX*/, 1/*HEIGHT_IDX*/);

    DMSG("%s C:%d\n", __func__, C);

    softmax(input, C, beta, output);
    return;
}

static void shadownet_avgpool(unsigned cur_id, float *input, float *output); 
static void shadownet_avgpool(unsigned cur_id, float *input, float *output) {
    int H, W, C;
    int i,k;
    int out_index,in_index;

    // input tensor shape (batch, h, w, c)
    H = get_operator_tensor_shape_dim(cur_id, 0/*INPUT_IDX*/, 1/*HEIGHT_IDX*/);
    W = get_operator_tensor_shape_dim(cur_id, 0/*INPUT_IDX*/, 2/*WIDTH_IDX*/);
    C = get_operator_tensor_shape_dim(cur_id, 0/*INPUT_IDX*/, 3/*CHANNEL_IDX*/);

    DMSG("%s H:%d, W:%d, C:%d\n", __func__, H, W, C);

    for(k = 0; k < C; ++k){
        out_index = k;
        output[out_index] = 0;
        for(i = 0; i < H*W; ++i){
            in_index = k + C*i; 
            output[out_index] += input[in_index];
        }
        output[out_index] /= H*W;
    }
}


// linear_transform_generic does linear_transform and add bias 
static void shadownet_linear_transform_generic(unsigned cur_id, float *input, float *output); 
static void shadownet_linear_transform_generic(unsigned cur_id, float *input, float *output) {
    int i,j, H, W, C;
    float *bias;

    DMSG("%s cur_id:%d, input:%p, output:%p\n", __func__, cur_id, input, output);

    // input tensor shape (batch, h, w, c)
    H = get_operator_tensor_shape_dim(cur_id, 0/*INPUT_IDX*/, 1/*HEIGHT_IDX*/);
    W = get_operator_tensor_shape_dim(cur_id, 0/*INPUT_IDX*/, 2/*WIDTH_IDX*/);

    // weight tensor shape (2, units)
    C = get_operator_tensor_shape_dim(cur_id, LT_WEIGHT1_IDX, UNIT_IDX);

    bias = (float *)get_operator_tensor_buffer(cur_id, LT_WEIGHT3_IDX, true); 

    shadownet_linear_transform(cur_id, input, output);

    for(j = 0; j < H*W; ++j)
        for(i = 0; i < C; ++i)
            output[j*C + i] = output[j*C + i] + bias[i];
}

static void shadownet_dense(unsigned cur_id, float *input, float *output);
static void shadownet_dense(unsigned cur_id, float *input, float *output)
{
    int B, H, R, C;
    int i, j, k;
    float *weights;
    float *bias, t;

    // input tensor shape (b, h), weights: (r, c)
    B = get_operator_tensor_shape_dim(cur_id, 0/*INPUT_IDX*/, 0/*BATCH_IDX*/);
    H = get_operator_tensor_shape_dim(cur_id, 0/*INPUT_IDX*/, 1/*HEIGHT_IDX*/);
    R = get_operator_tensor_shape_dim(cur_id, 1/*ROW_IDX*/, 0/*CHANNEL_IDX*/);
    C = get_operator_tensor_shape_dim(cur_id, 1/*COL_IDX*/, 1/*CHANNEL_IDX*/);

    weights = (float *)get_operator_tensor_buffer(cur_id, 1/*WEIGHT_IDX*/, true/*is_float*/); 
    bias= (float *)get_operator_tensor_buffer(cur_id, 2/*WEIGHT_IDX*/, true /*is_float*/); 

    DMSG("%s B:%d, H:%d, R:%d, C:%d, weights:%p, bias:%p\n", __func__, B, H, R, C, weights, bias);

    for (i = 0; i < B; i++)
        for (j=0; j < C; j++) {
            t = 0.0;
            for (k = 0; k < H; k++) {
               t += input[i*H + k] * weights[k*C+j];
            }
            t += bias[k];
            output[i * C + j] = t;
        }
    return;
}

static void shadownet_shuffle_channel(unsigned cur_id, float *input, float *output); 
static void shadownet_shuffle_channel(unsigned cur_id, float *input, float *output) {
    int H, W, C;
    int h, w, c, idx_from;
    float scalar;
    int *sf_obfweights;
    float *sf_rbias;

    // input tensor shape (batch, h, w, c)
    H = get_operator_tensor_shape_dim(cur_id, 0/*INPUT_IDX*/, 1/*HEIGHT_IDX*/);
    W = get_operator_tensor_shape_dim(cur_id, 0/*INPUT_IDX*/, 2/*WIDTH_IDX*/);
    C = get_operator_tensor_shape_dim(cur_id, 0/*INPUT_IDX*/, 3/*CHANNEL_IDX*/);

    sf_obfweights = (int *)get_operator_tensor_buffer(cur_id, 1/*WEIGHT_IDX*/, false /*is_float*/); 
    sf_rbias= (float *)get_operator_tensor_buffer(cur_id, 2/*WEIGHT_IDX*/, true /*is_float*/); 

    DMSG("%s H:%d, W:%d, C:%d, weights1:%p, weights2:%p\n", __func__, H, W, C, sf_obfweights, sf_rbias);
    assert(sf_obfweights!= NULL);
    assert(sf_rbias!= NULL);

    for (h = 0; h < H; ++h) {
        for (w = 0; w < W; ++w) {
            for (c = 0; c < C; ++c) {
                idx_from = sf_obfweights[c];
                scalar = sf_rbias[c];

                output[(h * W * C) + (w * C) + c] = 
                  input[(h * W * C) + (w * C) + idx_from] * scalar;
            }
        }
    }
}

static void handle_custom_op(unsigned op_id, float *input, float *output); 
static void handle_custom_op(unsigned op_id, float *input, float *output) {
    unsigned opcode_id;
    flatbuffers_string_t op_name; 
    opcode_id = ns(Operator_opcode_index(ns(Operator_vec_at(model_operators, op_id))));
    op_name = ns(OperatorCode_custom_code(ns(OperatorCode_vec_at(model_opcodes, opcode_id))));

    DMSG("%s op_name:%s\n", __func__, op_name);

    if (strcmp(op_name, "LinearTransform") == 0) {
        shadownet_linear_transform(op_id, input, output);
    } else if (strcmp(op_name, "LinearTransformGeneric") == 0) {
        shadownet_linear_transform_generic(op_id, input, output);
    } else if (strcmp(op_name, "AddMask") == 0) {
        shadownet_add_mask(op_id, input, output);
    } else if (strcmp(op_name, "ShuffleChannel") == 0) {
        shadownet_shuffle_channel(op_id, input, output);
    } else {
       DMSG("ERROR %s: custom_op %s not recognized!\n", __func__, op_name); 
    } 
}

static unsigned get_op_output_size(unsigned op_id); 
static unsigned get_op_output_size(unsigned op_id) {
    unsigned output_size, tid;
    size_t out;
    flatbuffers_int32_vec_t shape;
    size_t shape_len;
    flatbuffers_int32_vec_t outputs;
    outputs = ns(Operator_outputs(ns(Operator_vec_at(model_operators, op_id))));
    tid = flatbuffers_int32_vec_at(outputs, 0);
    shape = ns(Tensor_shape(ns(Tensor_vec_at(model_tensors, tid))));
    shape_len = flatbuffers_int32_vec_len(shape);
    //printf("tensor shape: ");
    output_size = 1;
    for (int j = 0; j < shape_len; j ++) {
        printf(" %d, ", flatbuffers_int32_vec_at(shape, j));
        output_size *= flatbuffers_int32_vec_at(shape, j);
    }
    return output_size * 4; /*bytes*/

}


 // To save memory, we rotate the use of output buffer
// We allocate two big buffer, outbuf1, outbuf2
// (first layer)   local id 0: inout, weights, outbuf1
// (middle layers) local id 1: outbuf1, weights, outbuf2
// ...
// (last layer)    local id m: outbuf1/2, weights, inout
static void handle_operator(unsigned op_id, unsigned op_code, bool is_first, bool is_last, unsigned local_id, float* inbuf, float *outbuf, bool use_buf, bool store_buf); 
static void handle_operator(unsigned op_id, unsigned op_code, bool is_first, bool is_last, unsigned local_id, float* inbuf, float *outbuf, bool use_buf, bool store_buf) {
    float *input, *output;

    DMSG("%s op_id:%d", __func__, op_id);
    input = (local_id % 2 == 0) ? outbuf2: outbuf1;
    output = (local_id % 2 == 0) ? outbuf1: outbuf2;

    if (is_first)
        input = inbuf;
    else if (is_last) 
        output = outbuf;

    switch(op_code) {
        case CUSTOM:
            handle_custom_op(op_id, input, output);  
            break;
        case MUL:
#ifdef OPTIMIZE_MUL_ADD
            shadownet_mul_add(op_id, input, output, use_buf, store_buf);  
#else
            shadownet_mul(op_id, input, output);  
#endif

            break;
        case ADD: 
            shadownet_add(op_id, input, output, use_buf, store_buf);
            break;
        case MEAN:
        case AVERAGE_POOL_2D:
            shadownet_avgpool(op_id, input, output);
            break;
        case SOFTMAX:
            shadownet_softmax(op_id, input, output);
            break;
        case FULLY_CONNECTED:
            shadownet_dense(op_id, input, output);  
            break;
        default:
            DMSG("ERROR! %s: unhandled operator code : %d\n", __func__, op_code);
    }
}


static void handle_operator_multinputs(unsigned op_id, unsigned op_code, bool is_first, bool is_last, unsigned local_id, int num_inputs, float* inbuf[], float *outbuf, bool use_buf, bool store_buf); 
static void handle_operator_multinputs(unsigned op_id, unsigned op_code, bool is_first, bool is_last, unsigned local_id, int num_inputs, float* inbuf[], float *outbuf, bool use_buf, bool store_buf) {
    float *input, *output;

    DMSG("%s op_id:%d", __func__, op_id);
    input = (local_id % 2 == 0) ? outbuf2: outbuf1;
    output = (local_id % 2 == 0) ? outbuf1: outbuf2;

    if (is_first)
        input = inbuf;
    else if (is_last) 
        output = outbuf;

    switch(op_code) {
        case CUSTOM:
            handle_custom_op(op_id, input, output);  
            break;
        case MUL:
#ifdef OPTIMIZE_MUL_ADD
            shadownet_mul_add(op_id, input, output, use_buf, store_buf);  
#else
            shadownet_mul(op_id, input, output);  
#endif

            break;
        case ADD: 
            shadownet_add(op_id, input, output, use_buf, store_buf);
            break;
        case MEAN:
        case AVERAGE_POOL_2D:
            shadownet_avgpool(op_id, input, output);
            break;
        case SOFTMAX:
            shadownet_softmax(op_id, input, output);
            break;
        case FULLY_CONNECTED:
            shadownet_dense(op_id, input, output);  
            break;
        default:
            DMSG("ERROR! %s: unhandled operator code : %d\n", __func__, op_code);
    }
}

// op_id :the current linear operator id in Normal World
//
// Note: shadownet start from op_id + 1, and end until 
// the next linear layer/PAD layer in Normal World
// or the end of network
void shadownet_inference(unsigned op_id, float *input, float *output) {
    unsigned cur_id, local_id;
    unsigned op_code, next_op_code;
    unsigned MAX_OP_ID = get_max_op_id();
    bool is_last, is_first;

    local_id = 0; // local id for the following non-linear layers 
    cur_id = op_id + 1;

    op_code = get_op_code(cur_id);
    next_op_code = get_op_code(cur_id+1);

    DMSG("shadownet_inference input: %p, output:%p\n", input, output);

    while (op_code != CONV_2D && op_code != DEPTHWISE_CONV_2D && op_code != PAD) {
        DMSG("handle operator %d, op_code:%d, local_id:%d, max_op_id:%d\n",cur_id, op_code, local_id, MAX_OP_ID);

        is_last = false;
        is_first = false;

        // skip reshape
        if(op_code == RESHAPE) {
            DMSG("Skip Reshape Op as nothing needs to be done!\n");
            cur_id += 1;
            op_code = get_op_code(cur_id);
            next_op_code = (cur_id != MAX_OP_ID) ? get_op_code(cur_id+1): 0;
            continue;
        }

        if (cur_id == MAX_OP_ID || next_op_code == CONV_2D || next_op_code == DEPTHWISE_CONV_2D)
            is_last = true;

        if (local_id == 0)
            is_first = true;

        handle_operator(cur_id, op_code, is_first, is_last, local_id, input, output, false, false);

        if (is_last)
            break;

        local_id += 1;
        cur_id += 1;

#ifdef OPTIMIZE_MUL_ADD
        // skip the ADD following MUL
        if (op_code == MUL) {
            local_id += 1;
            cur_id += 1;
        }
#endif

        op_code = get_op_code(cur_id);
        next_op_code = (cur_id != MAX_OP_ID) ? get_op_code(cur_id+1): 0;
    }
}

void shadownet_inference_multinputs(unsigned op_id, int num_inputs, float *inputs[], float *output) {
    unsigned cur_id, local_id;
    unsigned op_code, next_op_code;
    unsigned MAX_OP_ID = get_max_op_id();
    bool is_last, is_first;

    local_id = 0; // local id for the following non-linear layers 
    cur_id = op_id + 1;

    op_code = get_op_code(cur_id);
    next_op_code = get_op_code(cur_id+1);

    DMSG("shadownet_inference input: %p, output:%p\n", inputs, output);

    while (op_code != CONV_2D && op_code != DEPTHWISE_CONV_2D && op_code != PAD) {
        DMSG("handle operator %d, op_code:%d, local_id:%d, max_op_id:%d\n",cur_id, op_code, local_id, MAX_OP_ID);

        is_last = false;
        is_first = false;

        // skip reshape
        if(op_code == RESHAPE) {
            DMSG("Skip Reshape Op as nothing needs to be done!\n");
            cur_id += 1;
            op_code = get_op_code(cur_id);
            next_op_code = (cur_id != MAX_OP_ID) ? get_op_code(cur_id+1): 0;
            continue;
        }

        if (cur_id == MAX_OP_ID || next_op_code == CONV_2D || next_op_code == DEPTHWISE_CONV_2D)
            is_last = true;

        if (local_id == 0)
            is_first = true;

        handle_operator_multinputs(cur_id, op_code, is_first, is_last, local_id,num_inputs, inputs, output, false, false);

        if (is_last)
            break;

        local_id += 1;
        cur_id += 1;

#ifdef OPTIMIZE_MUL_ADD
        // skip the ADD following MUL
        if (op_code == MUL) {
            local_id += 1;
            cur_id += 1;
        }
#endif

        op_code = get_op_code(cur_id);
        next_op_code = (cur_id != MAX_OP_ID) ? get_op_code(cur_id+1): 0;
    }
}


void resnet_inference(unsigned plan_id, float *input, float *output) {
    int node_num = resnet_plans[plan_id].node_num;
    for (int i = 0; i < node_num; ) {
#ifdef OPTIMIZE_MUL_ADD
        // when see MUL, do MUL_ADD and skip ADD
        if (resnet_plans[plan_id].nodes[i].op_code == MUL) { 
            handle_operator(resnet_plans[plan_id].nodes[i].op_id,
                        resnet_plans[plan_id].nodes[i].op_code,
                        i == 0/*is_first*/,
                        i == node_num-1/*is_last*/,
                        i/*local_id*/,
                        input,
                        output,
                        resnet_plans[plan_id].nodes[i].read_buf,
                        /* check the following ADD, need write buf? */
                        resnet_plans[plan_id].nodes[i+1].write_buf);
            i++;
        } else {
            handle_operator(resnet_plans[plan_id].nodes[i].op_id,
                        resnet_plans[plan_id].nodes[i].op_code,
                        i == 0/*is_first*/,
                        i == node_num-1/*is_last*/,
                        i/*local_id*/,
                        input,
                        output,
                        resnet_plans[plan_id].nodes[i].read_buf,
                        resnet_plans[plan_id].nodes[i].write_buf);
        }

#else

        handle_operator(resnet_plans[plan_id].nodes[i].op_id,
                    resnet_plans[plan_id].nodes[i].op_code,
                    i == 0/*is_first*/,
                    i == node_num-1/*is_last*/,
                    i/*local_id*/,
                    input,
                    output,
                    resnet_plans[plan_id].nodes[i].read_buf,
                    resnet_plans[plan_id].nodes[i].write_buf);
#endif

        i++;
    }
}

// return number of elements in mask weights
static unsigned get_single_mask_size(unsigned cur_id); 
static unsigned get_single_mask_size(unsigned cur_id) {
    unsigned H, W, C;
    // input tensor shape (batch, h, w, c)
    H = get_operator_tensor_shape_dim(cur_id, 0/*INPUT_IDX*/, 1/*HEIGHT_IDX*/);
    W = get_operator_tensor_shape_dim(cur_id, 0/*INPUT_IDX*/, 2/*WIDTH_IDX*/);
    C = get_operator_tensor_shape_dim(cur_id, 0/*INPUT_IDX*/, 3/*CHANNEL_IDX*/);
    
    return H*W*C*4;
}

static bool is_mask_layer(unsigned op_id); 
static bool is_mask_layer(unsigned op_id) {
    unsigned opcode_id, op_code;
    char *op_name; 
    op_code = get_op_code(op_id);
    if (op_code == CUSTOM) { 
        opcode_id = ns(Operator_opcode_index(ns(Operator_vec_at(model_operators, op_id))));
        op_name = ns(OperatorCode_custom_code(ns(OperatorCode_vec_at(model_opcodes, opcode_id))));
        if(strcmp(op_name, "AddMask") == 0)
            return true; 
    } 

    return false;
}

// update model's masks with given masks from buffer masks
int update_masks(uint8_t *masks, unsigned mask_size)
{
    size_t num = ns(Operator_vec_len(model_operators));
    unsigned ms, offset = 0;
    uint8_t *add_masks;

    if (num <=0) return -1;
    for (int i = 0; i < num; i++) {
        if (is_mask_layer(i)) {
            ms = get_single_mask_size(i); // in bytes
            add_masks = (uint8_t *)get_operator_tensor_buffer(i, 1/*WEIGHT_IDX*/, true /*is_float*/); 
            memcpy(add_masks, masks + offset, ms);
            offset += ms;
        } else
            continue;
    }

    assert(mask_size == offset);

    return 0;
}

// we can design a stream based mask updating mechanism
// so that we can update partial masks for each round of 
// mask loading. The key point is to keep track of where
// we are, each time we get a new chunk of masks.
// this will save the extra memory allocated for storing the masks
// temperally
int fake_update_masks(uint8_t *masks, unsigned mask_len) {
    //TEE_MemMove(masks, masks, mask_size);
    for (int i = 0; i < mask_len; i++)
        masks[i] = masks[0];
    return 0;
}
