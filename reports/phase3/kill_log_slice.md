# Phase 3 gate — kill-log slice

Every kill is an append-only row with an `undone` column. The full kill set is
recoverable at any time as ledger minus survivors — nothing is deleted.


## `stopword_straddler` — 458,293 killed

```
  and
  to
  a
  of
  in
  we
  for
  this
  on
  that
  with
  by
  our
  as
  from
  an
  of the
  which
  in this
  we propose
```

## `verb_led` — 127,051 killed

```
  be
  propose
  use
  can
  show
  demonstrate
  achieve
  introduce
  base
  propose a
  show that
  exist
  improve
  learn
  present
  base on
  provide
  address
  generate
  train
```

## `non_nominal_head` — 53,336 killed

```
  novel
  large
  different
  new
  neural
  available
  deep
  extensive
  high
  recent
  human
  multiple
  far
  significant
  visual
  significantly
  semantic
  experimental
  specifically
  effective
```

## `ordinal` — 2,121 killed

```
  first
  previous
  second
  previous work
  next
  previous method
  last
  latter
  third
  first time
  previous state-of-the-art
  former
  previous study
  previous approach
  first to
  first step
  first stage
  second stage
  first propose
  first introduce
```

## `generic_head` — 633 killed

```
  model
  method
  dataset
  result
  performance
  task
  approach
  paper
  experiment
  work
  feature
  framework
  information
  study
  accuracy
  problem
  system
  application
  analysis
  evaluation
```
