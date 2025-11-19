FROM rust:1 AS chef

RUN cargo install cargo-chef
WORKDIR /app

FROM chef AS planner

COPY . .
RUN cargo chef prepare  --recipe-path recipe.json

FROM chef AS builder

RUN apt update
RUN apt install -y pkg-config ca-certificates build-essential
RUN apt install -y cmake clang

COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json
COPY . .
RUN cargo build --release --bin ortts

FROM bitnami/minideb AS runtime
WORKDIR /app
COPY --from=builder /app/target/release/ortts /usr/local/bin

EXPOSE 12775
ENTRYPOINT ["/usr/local/bin/ortts"]
CMD ["serve", "--listen", "0.0.0.0:12775"]
