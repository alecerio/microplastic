CC = gcc
CFLAGS = -I$(INC)

SRC = src
INC = include
BUILD = build
OUT = out

TARGET = quant

SRC_QUANT = $(SRC)/main.c $(SRC)/quant_helper.c
OBJ_QUANT = $(patsubst $(SRC)/%.c,$(BUILD)/%.o,$(SRC_QUANT))

all: $(TARGET)

quant: $(OBJ_QUANT)
	$(CC) -o $(OUT)/$@ $^ 

$(BUILD)/%.o: $(SRC)/%.c
	$(CC) -c $(CFLAGS) -o $@ $<

clean:
	rm -f $(BUILD)/*
	rm -f $(OUT)/*

