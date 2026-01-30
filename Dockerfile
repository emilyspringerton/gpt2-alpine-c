\
    FROM alpine:latest

    RUN apk add --no-cache gcc musl-dev make

    WORKDIR /app

    COPY . /app

    RUN make

    CMD ["./gpt2_run", "weights/model.bin"]
