# running ortts.
run *args:
  bacon run -- {{args}}

dev *args:
  bacon run -- serve {{args}}

serve *args:
  bacon run -- serve {{args}}

# lint code. (args example: just lint --fix)
lint *args:
  cargo clippy {{args}} -- -W clippy::pedantic -W clippy::nursery
