#ifndef PARAMETERS_H_
#define PARAMETERS_H_

//#define FP
#define INT

typedef short data_t; // data in words

//#define data_t short

#ifdef FP
// potential stages in floating point DSP slice
#define FMA_LATENCY		7
#define MAC_LATENCY		(FMA_LATENCY - 1)
#endif

#ifdef INT
// stages in DSP48E1 slice
#define FMA_LATENCY 	4 // add extra pipeline register for LAPU PE inout
#define MAC_LATENCY 	(FMA_LATENCY - 1)
#endif

// size of scratch regs for each LAC
#define SCRATCH_SIZE 	4

// number of LACs per LAP
#define LAC_NUM 		1

// kernel size of rank-k updates
// at least twice the size of Nr
// in order to prefetch and postpush C
// as well as read in the new B panel
#define Kc 				64

// number of rows in the A panel
// determines the number of rank-k updates performed
// before needing a new A panel
#define Mc 				64

// Nr x Nr PEs per LAC
#define Nr				4

// max size
// size of PE local memory
// rows x cols of A divided by num PEs
#define MEM_A_SIZE		((Mc * Kc) / (Nr * Nr))

// max size
// size Kc due to B replication
// multiplied by two for space for B prefetch
#define MEM_B_SIZE		(Kc * 2)

// max size
// input matrix A, B
// M x K and K x N
#define MAX_M			256 // words
#define MAX_N			256 // words

const int FETCH_A_CYCLES = ((Mc * Kc) / Nr); // 4 PEs are loaded with an A panel value each cycle
const int FETCH_B_CYCLES = Kc; // B strips are size Kc x Nr and Nr PEs sharing a column bus load same B value each cycle
const int FETCH_C_CYCLES = Nr; // 4 PEs are loaded with their respective C value from the column bus each cycle
const int RANK_K_UPDATE_CYCLES = Kc; // Kc broadcasts in a rank-k update
const int PUSH_C_CYCLES = Nr; // 4 PEs broadcast their respective C value on the column bus each cycle
const int RANK_K_PER_RANK_M = (Mc / Nr);

enum Test_Matrix{

  DIAG,
  DIAG_INCR,
  DIAG_MULT,
  ALL0,
  ALL1,
  PRESET,
  UP_RIGHT_TRIU_PRESET,
  UP_LEFT_TRIU_PRESET,
  LO_TRIU_PRESET,
  BANDED,
  RM_ASCENDING,


/*
	DIAG                  : Diagonal with fixed preset
	DIAG_INCR             : Increasing diagonal with Preset
	DIAG_MULT             : Increasing diagonal with multiple of preset
	ALL0                  : All 0 matrix
	ALL1                  : All 1 matrix
  PRESET                : Constant Matrix with preset value
	UP_RIGHT_TRIU_PRESET  : Top Right aligned triangular with preset
	UP_LEFT_TRIU_PRESET   : Top Left aligned triangular with Preset
  LO_TRIU_PRESET        : Lower triangular with preset
  BANDED                : Banded Matrix
*/

};

enum Lac_Ctrl {
	CTRL_IDLE,
	CTRL_FETCH_A,
    CTRL_FETCH_B,
	CTRL_OPERATE,
	CTRL_RST
};

#endif /* PARAMETERS_H_ */
