CC = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra -Isrc
LDFLAGS = -lm
SRC = src/gpt2.c src/checkpoint_loader.c src/tokenizer.c src/archetype.c src/main.c
OBJ = $(SRC:.c=.o)
BIN = gpt2_run

all: $(BIN)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN): $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

weights/tokenizer.bin: weights/tokenizer.json scripts/build_tokenizer_bin.py
	python3 scripts/build_tokenizer_bin.py

tokenizer: weights/tokenizer.bin

# S150-01: Python↔Go interop point for towerprint-augmented training
# records (scripts/prime_directive_dataset.py --towerprint-augment).
bin/towerprint-cli: cmd/towerprint-cli/main.go pkg/towerprint/*.go
	@mkdir -p bin
	go build -o bin/towerprint-cli ./cmd/towerprint-cli

towerprint-cli: bin/towerprint-cli

test: $(BIN) bin/towerprint-cli
	@bash tests/test_compile.sh
	@python3 -m pytest tests/test_dataset.py tests/test_corpus_stats.py -v 2>/dev/null || (python3 tests/test_dataset.py && python3 tests/test_corpus_stats.py)
	@cd pkg/towerprint && go test ./...

clean:
	rm -f $(OBJ) $(BIN)
	rm -rf bin/

.PHONY: all clean test tokenizer towerprint-cli
