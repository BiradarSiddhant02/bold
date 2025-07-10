# === Compiler and Flags ===
CC := gcc
COMMON_CFLAGS := -O3 -Wall -Wextra -std=c11 -Iinclude -fPIC -fopenmp -funroll-loops
SO_FLAGS := -shared -fPIC -fopenmp
LDFLAGS := -lm -Wl,--version-script=exports.map

# === Directories ===
SRC_DIR := src
BIN_DIR := bin
OUT_DIR := ClusterIndex

# === Output Shared Library ===
SHARED_LIB := $(OUT_DIR)/libvecops.so

# === Ensure output dirs exist ===
$(shell mkdir -p $(BIN_DIR) $(OUT_DIR))

# === Detect Source Files ===
SRC_FILES := $(wildcard $(SRC_DIR)/*.c)

SSE_SRC     := $(filter %vecop_sse.c,     $(SRC_FILES))
AVX_SRC     := $(filter %vecop_avx.c,     $(SRC_FILES))
AVX2_SRC    := $(filter %vecop_avx2.c,    $(SRC_FILES))
AVX512_SRC  := $(filter %vecop_avx512.c,  $(SRC_FILES))
DISPATCH_SRC:= $(filter %dispatcher.c,$(SRC_FILES))
OTHER_SRC   := $(filter-out $(SSE_SRC) $(AVX_SRC) $(AVX2_SRC) $(AVX512_SRC) $(DISPATCH_SRC), $(SRC_FILES))

# === Object Files (placed in /bin) ===
SSE_OBJ      := $(BIN_DIR)/vecop_sse.o
AVX_OBJ      := $(BIN_DIR)/vecop_avx.o
AVX2_OBJ     := $(BIN_DIR)/vecop_avx2.o
AVX512_OBJ   := $(BIN_DIR)/vecop_avx512.o
DISPATCH_OBJ := $(BIN_DIR)/dispatcher.o
OTHER_OBJ    := $(patsubst $(SRC_DIR)/%.c,$(BIN_DIR)/%.o,$(OTHER_SRC))

OBJS := $(filter-out $(BIN_DIR)/.o,$(SSE_OBJ) $(AVX_OBJ) $(AVX2_OBJ) $(AVX512_OBJ) $(DISPATCH_OBJ) $(OTHER_OBJ))

# === Default Target ===
all: $(SHARED_LIB)

# === Compilation Rules ===

$(BIN_DIR)/vecop_sse.o: $(SRC_DIR)/vecop_sse.c
	$(CC) $(COMMON_CFLAGS) -msse2 -c $< -o $@

$(BIN_DIR)/vecop_avx.o: $(SRC_DIR)/vecop_avx.c
	$(CC) $(COMMON_CFLAGS) -mavx -c $< -o $@

$(BIN_DIR)/vecop_avx2.o: $(SRC_DIR)/vecop_avx2.c
	$(CC) $(COMMON_CFLAGS) -mavx2 -mfma -c $< -o $@

$(BIN_DIR)/vecop_avx512.o: $(SRC_DIR)/vecop_avx512.c
	$(CC) $(COMMON_CFLAGS) -mavx512f -c $< -o $@

$(BIN_DIR)/dispatcher.o: $(SRC_DIR)/dispatcher.c
	$(CC) $(COMMON_CFLAGS) -c $< -o $@

# Generic rule for other source files
$(BIN_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(COMMON_CFLAGS) -c $< -o $@

# === Link all object files into shared object ===
$(SHARED_LIB): $(OBJS)
	$(CC) $(SO_FLAGS) -o $@ $^ $(LDFLAGS)

# === Clean ===
clean:
	rm -f $(BIN_DIR)/*.o $(SHARED_LIB)
