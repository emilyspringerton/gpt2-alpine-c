CC = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra -lm
SRC = src/gpt2.c src/checkpoint_loader.c src/main.c
OBJ = $(SRC:.c=.o)
BIN = gpt2_run

all: $(BIN)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN): $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -f $(OBJ) $(BIN)

.PHONY: all clean
