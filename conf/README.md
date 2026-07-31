# Experiment configuration

YAML here describes *experiments*: which features and targets, which model and
hyperparameters, which evaluation weights. Runtime configuration — where the
data lives, which seed, how verbose — is separate and lives in
`vision_zero.config`.

The separation makes an experiment a reviewable artifact. A result can be traced
to the exact file that produced it, and reproducing it is a matter of passing
that file rather than reconstructing a set of values from context.

Each file is parsed into a Pydantic model, so a malformed experiment fails at
load with a precise path to the offending key rather than at fit time with a
`KeyError`.
