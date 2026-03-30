Kubeflow SDK OpenTelemetry; GSoC 2026 Project 7 Prototype
Author: Aarushi Verma (github.com/aashvgit)

Demonstrates:
  1. kubeflow/common/telemetry.py  — shared foundation module
  2. TrainerClient instrumentation  — spans, attributes, metrics
  3. PipelinesClient instrumentation — spans, attributes
  4. Cross-client trace propagation  — single connected trace
  5. Exception recording             — errors captured on spans
  6. Zero-overhead disabled mode     — no OTel cost when off

Run:
  pip install opentelemetry-api opentelemetry-sdk
  python kubeflow_otel_proto.py

LAYER 1: kubeflow/common/telemetry.py

  OTel imports (lazy — only loaded when enabled)
  configure_telemetry():
    Call once at startup to enable OpenTelemetry.
    Supports exporters: 'console', 'otlp', 'none'
  get_tracer(): Return active tracer or no-op tracer if OTel is disabled.
  get_meter(): Return active meter or no-op meter if OTel is disabled.
  trace_method():Decorator: wrap any SDK method in an OTel span.
    Records exceptions automatically and sets span status.
    Usage:
        @trace_method("TrainerClient.train")
        def train(self, ...):
            ...
  measure_duration():Decorator: record method execution time as an OTel histogram.
    Usage:
        @measure_duration("kubeflow.trainer.job.duration")
        def train(self, ...):
            ...


LAYER 2a: kubeflow/trainer/client.py
  Instrumented TrainerClient. Every public method emits an OTel span with semantic attributes.
  Metrics defined once per client instance
  train():Submit a distributed TrainJob. Returns job_id, Set semantic span attributes, GenAI semantic convention,
  Simulate job submission child span and Simulate polling child span 
  get_job(): Get status of a TrainJob by ID.
  train_with_error():Demonstrates exception recording on spans.

LAYER 2b: kubeflow/pipelines/client.py
  Instrumented PipelinesClient. Traces pipeline compilation and submission.
  compile_pipeline():Compile a KFP pipeline. Returns path to compiled YAML.
  submit_pipeline():Submit a compiled pipeline. Returns run_id.

LAYER 3: Zero-overhead disabled mode demo
  TrainerClientNoOTel(TrainerClient): Same client demonstrates zero overhead when OTel is off.

DEMO RUNNER
  demo_disabled_mode(): Show zero overhead when OTel is not configured. No configure_telemetry() called uses no op tracer, Decorator still wraps but no-op tracer = zero cost
  demo_trainer_client(): Demonstrate TrainerClient instrumentation.
  demo_pipelines_client(): Demonstrate PipelinesClient instrumentation.
  demo_cross_client_trace(): Demonstrate cross-client trace propagation.
  demo_exception_recording(): Demonstrate exception recording on spans.
  demo_overhead_benchmark(): Benchmark overhead of instrumented vs plain method.


To generate prototype output:
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc
python kubeflow_otel_proto.py 2>&1 | tee prototype_output.txt  
  







