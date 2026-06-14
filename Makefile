CC = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra -Isrc
LDFLAGS = -lm
SRC = src/gpt2.c src/checkpoint_loader.c src/tokenizer.c src/main.c
OBJ = $(SRC:.c=.o)
BIN = gpt2_run

all: $(BIN)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN): $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

test: $(BIN)
	@bash tests/test_compile.sh
	@python3 -m pytest tests/test_dataset.py -v 2>/dev/null || python3 tests/test_dataset.py

clean:
	rm -f $(OBJ) $(BIN)

.PHONY: all clean test
