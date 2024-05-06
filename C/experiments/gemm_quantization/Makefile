CC = gcc
CFLAGS = -I$(INC)

SRC = src
INC = include
BUILD = build
OUT = out

TARGET = quant qgemm qsoftmax qmlp

SRC_QUANT = $(SRC)/main.c $(SRC)/quant_helper.c
OBJ_QUANT = $(patsubst $(SRC)/%.c,$(BUILD)/%.o,$(SRC_QUANT))

SRC_QGEMM = $(SRC)/main_qgemm.c $(SRC)/quant_helper.c $(SRC)/qgemm.c
OBJ_QGEMM = $(patsubst $(SRC)/%.c,$(BUILD)/%.o,$(SRC_QGEMM))

SRC_QSOFTMAX = $(SRC)/main_qsoftmax.c $(SRC)/quant_helper.c $(SRC)/qsoftmax.c
OBJ_QSOFTMAX = $(patsubst $(SRC)/%.c,$(BUILD)/%.o,$(SRC_QSOFTMAX))

SRC_QMLP = $(SRC)/main_mlp.c $(SRC)/quant_helper.c $(SRC)/qgemm.c
OBJ_QMLP = $(patsubst $(SRC)/%.c,$(BUILD)/%.o,$(SRC_QMLP))

all: $(TARGET)

quant: $(OBJ_QUANT)
	$(CC) -o $(OUT)/$@ $^

qgemm: $(OBJ_QGEMM)
	$(CC) -o $(OUT)/$@ $^ 

qsoftmax: $(OBJ_QSOFTMAX)
	$(CC) -o $(OUT)/$@ $^ 

qmlp: $(OBJ_QMLP)
	$(CC) -o $(OUT)/$@ $^ -lm

$(BUILD)/%.o: $(SRC)/%.c
	$(CC) -c $(CFLAGS) -o $@ $<

clean:
	rm -f $(BUILD)/*
	rm -f $(OUT)/*

