# ml_framework/orchestration/v2/specs — one file per business module.
#
# Each file declares:
#   - invoke(ctx: PipelineContext) -> None   : thin wrapper re-using existing
#                                               business logic verbatim
#   - adapter(raw_output) -> Recommendation  : only for modules that produce
#                                               a decision signal
#   - SPEC: ModuleSpec                       : the declarative registration
#
# invoke() takes a PipelineContext (not the ArtifactRegistry) by deliberate
# choice for Phase 1 — see docs/orchestration_v2_architecture.md Phase 1.
# The Execution Engine (Phase 3) is responsible for bridging ctx <-> registry.
