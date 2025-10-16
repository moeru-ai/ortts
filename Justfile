# running ortts.
run *args:
  bacon run -- {{args}}

dev *args:
  bacon run -- serve {{args}}

serve *args:
  bacon run -- serve {{args}}
