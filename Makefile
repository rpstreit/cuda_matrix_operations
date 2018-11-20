#
# Bobby Streit
# Nov. 18 2018
#

NVCC=nvcc
BIN=bin
OBJ=obj
CFLAGS=-dc -g 
LDFLAGS=
SOURCE_DIR=src
INCLUDE_DIR=include

# add any new files to this list here
FILES=lu_decomposition.cu linearSysSolver.cu main.cu managed.cu matrix.cu matrix_operations.cu reductions.cu tests.cu cpu.cu matrix_inverse.cu
SOURCES=$(FILES:%=$(SOURCE_DIR)/%)
OBJECTS=$(FILES:%.cu=$(OBJ)/%.o)

EXEC=$(BIN)/matrix_ops

all: $(EXEC)

$(EXEC): $(OBJECTS) $(BIN)
	$(NVCC) $(LDFLAGS) -o $@ $(OBJECTS) -I$(INCLUDE_DIR)

$(OBJECTS): $(OBJ)/%.o: $(SOURCE_DIR)/%.cu $(OBJ)
	$(NVCC) $(CFLAGS) -c $< -o $@ -I$(INCLUDE_DIR)

$(OBJ):
	mkdir $@

$(BIN):
	mkdir $@

clean:
	rm -f $(OBJ)/*.o
	rm -f $(EXEC)
