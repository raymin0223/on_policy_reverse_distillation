# Validation status

The release is checked with a maintainer CPU suite covering the gradient transforms, loss normalization, reward scoring, Agent protocol handling, experiment configurations, checkpoint retention/resumption, data integrity and distributed evaluation. Four-process CPU tests cover global token-mean gradients and evaluation sharding. These checks use synthetic models or stub model inference where appropriate.

The recipe checks compose all 120 student and 37 teacher configurations. Public Python files parse, shell launchers pass syntax checks, and the release-file scan checks for credentials and machine-specific paths. Maintainer tests and release tooling are not needed to train or evaluate.

Logging checks read actual offline W&B history and summary records, including delayed validation, source-step alignment and best-checkpoint metrics. Uneven-token four-process checks cover the main actor diagnostics and OPRD projection statistics; diagnostics-enabled backward checks preserve the training gradients.

Bounded full-finetuning smoke runs on four B200 GPUs with small models and short responses were completed before the latest code updates, including OPRD and selected section scripts. **The updated public code has not yet been re-tested on GPUs.** GPU memory usage and long-run stability still require validation on four B200 GPUs with the intended model sizes, sequence lengths and full training horizons. The earlier smoke runs and current CPU checks do not establish these properties for the updated code.

Full-size, long-context training, the complete Agent GPU workflow and all paper benchmark scores have not been validated end to end with this release. Runtime and memory measurements depend on the models, sequence lengths and hardware used. See the [training protocol](protocol_notes.md) and [reproduction guide](reproduction.md) for the executable settings and run instructions.
