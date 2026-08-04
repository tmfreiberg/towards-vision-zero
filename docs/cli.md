# Command reference

Every command takes an experiment name, which is the stem of a file in
`conf/experiment/`, or a path to a YAML file elsewhere. Run `uv run tvz --help`
for the current list and `uv run tvz COMMAND --help` for one command's options.

The experiments shipped with the repository:

| Name | Records | What it is |
|---|---|---|
| `montreal_ternary_complete` | Montréal, at most one blank field | The headline configuration, used throughout the book |
| `montreal_ternary` | Montréal, unfiltered | The same without the blank-field cap, used to show what the cap costs |
| `pedestrian_injury` | Montréal, pedestrian collisions | The subproblem of [Which variables matter](https://tmfreiberg.github.io/towards-vision-zero/book/04-what-matters.html) |
| `explorer_quebec` | All of Quebec, four severity levels | Feeds the interactive explorer |
| `montreal_ternary_baseline` | Montréal | A deliberately trivial configuration, for comparison |

## Getting started

### `tvz data prepare`

Compile the raw SAAQ CSVs into a single Parquet file. The only command that
reads `data/raw/`; everything else reads its output. Run once after cloning.

```bash
uv run tvz data prepare
uv run tvz data prepare --force   # recompile even if the output exists
```

### `tvz info`

Show the resolved paths and whether the expected inputs are present. The first
thing to run when a command cannot find something.

### `tvz data describe`

Print the codebook entry for a variable: its definition in the SAAQ's own words,
its documented values, and the gloss for each.

```bash
uv run tvz data describe LIGHT
uv run tvz data describe LIGHT --lang fr
```

### `tvz version`

Print the installed package version.

## Looking at the data

### `tvz examples`

Show groups of records that agree on every recorded variable and whose outcomes
nonetheless differ. The concrete form of the argument in [How well could anything do?](https://tmfreiberg.github.io/towards-vision-zero/book/03-limits.html).

```bash
uv run tvz examples montreal_ternary_complete
uv run tvz examples montreal_ternary_complete --count 10 --min-records 50
uv run tvz examples montreal_ternary_complete --worst-case   # groups holding both extremes
```

| Option | Default | |
|---|---|---|
| `--count` | 5 | How many groups to show |
| `--min-records` | 20 | Smallest group to consider |
| `--lang` | `en` | `en` or `fr` for value labels |
| `--worst-case` | off | Only groups holding both the mildest and severest outcome |

### `tvz audit`

Check whether any variable settles the outcome rather than informing it. Two
tests: whether a value is followed by exactly one outcome, and whether a value
makes some outcome impossible. The second is what caught the pedestrian victim
indicator described in [What the data was hiding](https://tmfreiberg.github.io/towards-vision-zero/book/02-quality.html).

```bash
uv run tvz audit montreal_ternary_complete
uv run tvz audit montreal_ternary_complete --feature ACCDN_TYPE   # one variable in full
```

| Option | Default | |
|---|---|---|
| `--min-support` | 30 | Records a value needs before it is judged |
| `--feature` | | Show one variable's full breakdown |

### `tvz missingness`

Report what a blank field reveals about the outcome. Blanks here are not
missing at random, and this is how that was established.

## Measuring the limit

### `tvz ceiling`

Estimate how well any classifier could do with these variables, by counting how
often identically recorded collisions disagree.

```bash
uv run tvz ceiling montreal_ternary_complete
uv run tvz ceiling montreal_ternary_complete --permutations 100
```

| Option | Default | |
|---|---|---|
| `--permutations` | 0 | Shuffles for the null test; 0 skips it |

The permutation test matters more than it looks. Shuffling the outcomes leaves
the in-sample figure almost unchanged, which is how one learns that it was
measuring the number of distinct combinations rather than predictability.

### `tvz locality`

Ask whether collisions that resemble each other turn out alike. Three views:
outcome similarity against how many variables differ, what each variable does
when it alone differs, and what a whole block does when everything outside it is
held fixed.

```bash
uv run tvz locality montreal_ternary_complete
uv run tvz locality montreal_ternary_complete --pairs   # also two variables at a time
```

| Option | Default | |
|---|---|---|
| `--pairs` | off | Also examine differences of two variables at a time |
| `--top` | 8 | Rows to show at each end of a ranking |

`--pairs` adds 153 groupings on eighteen variables and takes a few minutes. The
plain form takes under a minute.

### `tvz resolution`

Trace what the variables resolve against how many records they cover, by
matching on fewer of them at a time.

| Option | Default | |
|---|---|---|
| `--samples` | 12 | Random subsets per size |

## Fitting models

### `tvz train`

Fit one experiment and report the fold sizes and feature importances.

### `tvz sweep`

Compare estimator families under identical preprocessing, folds and scoring. If
the limit is in the data rather than the model, they land in the same place.

```bash
uv run tvz sweep montreal_ternary_complete
uv run tvz sweep montreal_ternary_complete --curve   # also the learning curve
```

| Option | Default | |
|---|---|---|
| `--folds` | 3 | Cross-validation folds per family |
| `--subsample` | 60000 | Cap on training records, to keep it affordable |
| `--curve` | off | Also compute the learning curve |

### `tvz ablate`

Measure what each group of variables contributes over and above the rest. Two
figures per block: what it achieves alone, and what it adds to everything else.

| Option | Default | |
|---|---|---|
| `--folds` | 3 | Cross-validation folds per variant |
| `--subsample` | 60000 | Cap on training records |

### `tvz evaluate`

Score a fitted model on the held-out records, and write the result to
`artifacts/reports/`. The only command in the project that touches those
records.

```bash
uv run tvz evaluate montreal_ternary_complete --i-am-finished-exploring
```

The flag is required and is meant to be awkward to type. Running this makes
every subsequent decision informed by the held-out set, so it belongs at the end
of an analysis rather than in the middle. [How the numbers were produced](https://tmfreiberg.github.io/towards-vision-zero/book/a2-methods.html) sets out what that
discipline is worth and where it is imperfect.

## Publishing

### `tvz export`

Build the static payload the explorer page reads.

```bash
uv run tvz export explorer_quebec
uv run tvz export explorer_quebec --to app
```

| Option | Default | |
|---|---|---|
| `--to` | `app` | Directory to write the payload into |

Re-run this whenever the schema changes, since the payload encodes the variable
list and would otherwise serve a stale one.
