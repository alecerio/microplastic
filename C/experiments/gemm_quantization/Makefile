CC = gcc
CFLAGS = -I$(INC)

SRC = src
INC = include
BUILD = build
OUT = out

TARGET = quant qgemm

SRC_QUANT = $(SRC)/main.c $(SRC)/quant_helper.c
OBJ_QUANT = $(patsubst $(SRC)/%.c,$(BUILD)/%.o,$(SRC_QUANT))

SRC_QGEMM = $(SRC)/main_qgemm.c $(SRC)/quant_helper.c $(SRC)/qgemm.c
OBJ_QGEMM = $(patsubst $(SRC)/%.c,$(BUILD)/%.o,$(SRC_QGEMM))

all: $(TARGET)

quant: $(OBJ_QUANT)
	$(CC) -o $(OUT)/$@ $^

qgemm: $(OBJ_QGEMM)
	$(CC) -o $(OUT)/$@ $^ 

$(BUILD)/%.o: $(SRC)/%.c
	$(CC) -c $(CFLAGS) -o $@ $<

clean:
	rm -f $(BUILD)/*
	rm -f $(OUT)/*

