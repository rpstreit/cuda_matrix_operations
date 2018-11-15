
CC=g++
BIN=bin
OBJ=obj
CFLAGS= -Wall -g
LDFLAGS=

MAT_SOURCES=matrix_maker.cpp
MAT_OBJECTS=$(MAT_SOURCES:%.cpp=$(OBJ)/%.o)
MM_EXEC=$(BIN)/matrix_maker

$(MM_EXEC): $(MAT_OBJECTS) $(BIN)
	$(CC) $(LDFLAGS) -o $@ $(MAT_OBJECTS) 

$(MAT_OBJECTS): $(OBJ)/%.o: %.cpp $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ):
	mkdir $@

$(BIN):
	mkdir $@

clean:
	rm -f $(OBJ)/*.o
	rm -f $(MM_EXEC)
