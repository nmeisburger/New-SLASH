CXX	:= g++

SRC_DIR   :=  ./src
BUILD_DIR := 	./build

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

TARGET := slash.cpp
BINARY := $(TARGET:.cpp=)

INC_FLAGS := -I/usr/local/include
LIB_FLAGS := -L/usr/local/lib

# -Ofast all optimizations of O3 plus bonus items
# -DNDEBUG disables assertions
# -fopenmp enables openmp library
# -march=native generates code for the cpu compiling the program and preforms optimizations on that ISA
# -fPIC generates position independent code
# -ffast-math faster but less precise math
# -funroll-loops unrolls loops with a fixed number of iterations at compile time
# -ftree-vectorize enables vectorization
CXX_OPT_FLAGS := -std=c++14 -Ofast -DNDEBUG -fopenmp -march=native -fPIC \
						 			-ffast-math -funroll-loops -ftree-vectorize 

CXX_DBG_FLAGS := -g -Wall -Wextra -Werror

CXX_FLAGS := $(INC_FLAGS) $(LIB_FLAGS) $(CXX_OPT_FLAGS) $(CXX_DBG_FLAGS) -lmpi

$(BINARY) : $(BUILD_DIR) $(OBJS)
	$(CXX) $(CXX_FLAGS) $(TARGET) $(OBJS) -o $@ 

$(OBJS) : $(BUILD_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(BUILD_DIR): 
	@mkdir -p $(BUILD_DIR)

clean: 
	rm -rf build slash

.PHONY: clean
