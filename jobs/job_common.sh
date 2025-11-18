#!/usr/bin/env bash
set -euo pipefail

DEFAULT_SEEDS=(1 2 3 4 5)

run_experiment() {
  srun "${APPTAINER_RUN[@]}" "${PYTHON_BIN}" -m experiment "$@"
}

add_seed_args() {
  local -n arr=$1
  local seeds=("${DEFAULT_SEEDS[@]}")
  arr+=(--num-runs "${#seeds[@]}")
  arr+=(--seeds "${seeds[@]}")
}

add_gate_defaults() {
  additional_train_args+=(
    --use-gating
    --entropy-loss-weight 0.0
    --sparsity-loss-weight 1.0
    --skip-modules
    --increasing-threshold
    --start-thr-percentile 0.0
    --end-thr-percentile 0.2
    --num-increasing-steps 6800
  )
  additional_eval_args+=(--use-gating --skip-modules)
}

ensure_max_hours() {
  local -n args=$1
  local hours=$2

  for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ "${args[$i]}" == "--max-hours" ]]; then
      return
    fi
  done

  args+=(--max-hours "$hours")
}

run_job() {
  local setting="$1"
  local mode="$2" # cot or loglikelihood
  local job_id="$3"

  local experiment_name="${job_id}_${mode}"
  local model_name="meta-llama/Llama-3.2-1B"
  local dataset_cot="csqa_gsm8k_gen"
  local dataset_loglik="fineweb"
  local eval_metrics_cot="csqa_gen,gsm8k,mmlu_gen,piqa_gen"
  local eval_metrics_loglik="commonsense_qa,piqa,mmlu_stem,hellaswag,winogrande,openbookqa,squadv2,lambada"
  local eval_batch_size=128
  local seq_length_train_cot=4096
  local seq_length_train_loglik=512
  local seq_length_eval=4096
  local batch_size=1
  local eval_train_batch_size=16
  local learning_rate=3e-6
  local warmup_steps=1000
  local lr_decay_steps=6800
  local max_epochs=1
  local finetune_mode="full"
  local enable_logging=true
  local save_checkpoint="${experiment_name}"
  local load_checkpoint="${experiment_name}"
  local additional_train_args=()
  local additional_eval_args=()
  local additional_eval_invocations=()
  local dataset
  local seq_length_train
  local eval_metrics

  if [[ "$mode" == "cot" ]]; then
    dataset="$dataset_cot"
    seq_length_train="$seq_length_train_cot"
    eval_metrics="$eval_metrics_cot"
  else
    dataset="$dataset_loglik"
    seq_length_train="$seq_length_train_loglik"
    eval_metrics="$eval_metrics_loglik"
  fi

  case "$setting" in
    MoD)
      additional_train_args+=(--use-mod --mod-capacity-factor 0.8 --skip-threshold 0.0)
      additional_eval_args+=(--use-mod --mod-capacity-factor 0.8 --skip-threshold 0.0)
      ;;
    CALM)
      additional_train_args+=(--use-early-exit --confidence-measure hidden_state)
      additional_eval_args+=(--use-early-exit --confidence-measure hidden_state)
      additional_eval_invocations+=("--confidence-measure softmax")
      ;;
    FREE)
      additional_train_args+=(
        --use-early-exit
        --early-exit-method free
        --use-free-distillation
        --use-free-speculative-decoding
        --free-shallow-layers 0 6
        --confidence-measure hidden_state
      )
      additional_eval_args+=(
        --use-early-exit
        --early-exit-method free
        --use-free-distillation
        --use-free-speculative-decoding
        --free-shallow-layers 0 6
        --confidence-measure hidden_state
      )
      additional_eval_invocations+=("--confidence-measure softmax")
      ;;
    LayerSkip)
      additional_train_args+=(
        --use-early-exit
        --early-exit-method layerskip
        --layer-skip-dropout 0.1
        --free-shallow-layers 0 6
      )
      additional_eval_args+=(
        --use-early-exit
        --early-exit-method layerskip
        --layer-skip-dropout 0.0
        --free-shallow-layers 0 6
      )
      ;;
    SkipLayer)
      additional_train_args+=(--use-skip-layer --skip-threshold 0.0)
      additional_eval_args+=(--use-skip-layer --skip-threshold 0.0)
      ;;
    random-skipping-baseline)
      add_gate_defaults
      additional_train_args+=(
        --no-one-gate-per-layer
        --randomly-skip
        --no-actually-gate
      )
      additional_eval_args+=(
        --no-one-gate-per-layer
        --randomly-skip
        --no-actually-gate
      )
      ;;
    random-skipping-baseline-pruning)
      add_gate_defaults
      additional_train_args+=(
        --no-one-gate-per-layer
        --randomly-skip
        --no-actually-gate
      )
      additional_eval_args+=(
        --no-one-gate-per-layer
        --randomly-skip
        --no-actually-gate
        --skip-layers
      )
      ;;
    random-skipping-baseline-wmt)
      add_gate_defaults
      dataset="wmt16"
      if [[ "$mode" == "cot" ]]; then
        eval_metrics="wmt_gen,mmlu_gen,piqa_gen"
      else
        eval_metrics="wmt_gen,lambada"
      fi
      eval_batch_size=64
      additional_train_args+=(
        --no-one-gate-per-layer
        --randomly-skip
        --no-actually-gate
        --max-hours 1
      )
      additional_eval_args+=(
        --no-one-gate-per-layer
        --randomly-skip
        --no-actually-gate
      )
      ;;
    GateSkip-separate-vector)
      add_gate_defaults
      ;;
    GateSkip-separate-scalar)
      add_gate_defaults
      additional_train_args+=(--single-number-gates)
      additional_eval_args+=(--single-number-gates)
      ;;
    GateSkip-shared-vector)
      add_gate_defaults
      additional_train_args+=(--no-one-gate-per-layer)
      additional_eval_args+=(--no-one-gate-per-layer)
      ;;
    GateSkip-only-attention)
      add_gate_defaults
      additional_eval_args+=(--skip-module-types attn)
      ;;
    GateSkip-only-mlp)
      add_gate_defaults
      additional_eval_args+=(--skip-module-types mlp)
      ;;
    GateSkip-entire-layer)
      add_gate_defaults
      additional_eval_args+=(--skip-layers)
      ;;
    GateSkip-every-second-layer)
      add_gate_defaults
      additional_eval_args+=(--only-skip-every-second-layer)
      ;;
    GateSkip-MLP)
      add_gate_defaults
      additional_train_args+=(--use-mlp-gate)
      additional_eval_args+=(--use-mlp-gate)
      ;;
    GateSkip-frozen-backbone-MLP)
      add_gate_defaults
      batch_size=2
      eval_train_batch_size=4
      seq_length_train_cot=2048
      seq_length_train_loglik=2048
      seq_length_train="$seq_length_train_cot"
      seq_length_eval=2048
      learning_rate=1e-4
      max_epochs=3
      finetune_mode="frozen"
      eval_batch_size=8
      additional_train_args+=(--use-mlp-gate --max-hours 9)
      additional_eval_args+=(--use-mlp-gate --max-hours 3)
      ;;
    GateSkip-frozen-backbone-MLP-shared)
      add_gate_defaults
      batch_size=2
      eval_train_batch_size=4
      seq_length_train_cot=2048
      seq_length_train_loglik=2048
      seq_length_train="$seq_length_train_cot"
      seq_length_eval=2048
      learning_rate=1e-4
      max_epochs=3
      finetune_mode="frozen"
      eval_batch_size=8
      additional_train_args+=(--use-mlp-gate --no-one-gate-per-layer --max-hours 9)
      additional_eval_args+=(--use-mlp-gate --no-one-gate-per-layer --max-hours 3)
      ;;
    GateSkip-before-module)
      add_gate_defaults
      additional_train_args+=(--gating-mode before_module)
      additional_eval_args+=(--gating-mode before_module)
      ;;
    GateSkip-L1-gate-loss)
      add_gate_defaults
      additional_train_args+=(--sparsity-loss-type l1)
      additional_eval_args+=(--sparsity-loss-type l1)
      ;;
    GateSkip-KL-gate-loss)
      add_gate_defaults
      additional_train_args+=(--sparsity-loss-type kl)
      additional_eval_args+=(--sparsity-loss-type kl)
      ;;
    GateSkip-frozen-backbone)
      add_gate_defaults
      batch_size=2
      eval_train_batch_size=4
      seq_length_train_cot=2048
      seq_length_train_loglik=2048
      seq_length_train="$seq_length_train_cot"
      seq_length_eval=2048
      learning_rate=1e-4
      max_epochs=5
      finetune_mode="frozen"
      eval_batch_size=8
      additional_train_args+=(--max-hours 9)
      additional_eval_args+=(--max-hours 3)
      ;;
    GateSkip-speculative-decoding)
      add_gate_defaults
      additional_eval_args+=(--use-self-speculative-decoding --self-speculative-budget 0.2)
      ;;
    GateSkip-pruning)
      add_gate_defaults
      additional_eval_args+=(--skip-layers --skip-entire-layer-based-on-attn)
      ;;
    GateSkip-quantization)
      add_gate_defaults
      additional_eval_args+=(--skip-layers)
      eval_batch_size=32
      additional_eval_invocations+=("--use-quantization")
      ;;
    GateSkip-wmt)
      add_gate_defaults
      dataset="wmt16"
      if [[ "$mode" == "cot" ]]; then
        additional_train_args+=(--max-hours 1)
        eval_metrics="wmt_gen,mmlu_gen,piqa_gen"
      else
        eval_metrics="wmt_gen,lambada"
      fi
      eval_batch_size=64
      additional_eval_args+=(--skip-layers)
      ;;
    GateSkip-Llama-3b)
      add_gate_defaults
      model_name="meta-llama/Llama-3.2-3B"
      eval_batch_size=32
      additional_eval_args+=(--skip-layers)
      ;;
    GateSkip-Llama-8b)
      add_gate_defaults
      model_name="meta-llama/Llama-3.1-8B"
      eval_batch_size=32
      additional_eval_args+=(--skip-layers)
      additional_train_args+=(--max-hours 8)
      ;;
    GateSkip-Gemma-2b)
      add_gate_defaults
      model_name="google/gemma-2-2b"
      eval_batch_size=32
      additional_eval_args+=(--skip-layers)
      ;;
    GateSkip-Llama-3b-Instruct)
      add_gate_defaults
      model_name="meta-llama/Llama-3.2-3B-Instruct"
      eval_batch_size=32
      additional_eval_args+=(--skip-layers)
      ;;
    *)
      echo "Unknown setting: $setting" >&2
      return 1
      ;;
  esac

  if [[ "$mode" == "loglikelihood" ]]; then
    ensure_max_hours additional_train_args 2
  fi

  if [[ "$mode" == "cot" ]]; then
    seq_length_train="$seq_length_train_cot"
  else
    seq_length_train="$seq_length_train_loglik"
  fi

  local train_args=(
    --experiment-name "$experiment_name"
    --model-name "$model_name"
    --dataset "$dataset"
    --pretrained
    --batch-size "$batch_size"
    --eval-batch-size "$eval_train_batch_size"
    --seq-length "$seq_length_train"
    --learning-rate "$learning_rate"
    --warmup-steps "$warmup_steps"
    --lr-decay-steps "$lr_decay_steps"
    --max-epochs "$max_epochs"
    --finetune-mode "$finetune_mode"
    --save-to-checkpoint "$save_checkpoint"
    --project-name gateskip
    "${additional_train_args[@]}"
  )

  local eval_args=(
    --experiment-name "$experiment_name"
    --model-name "$model_name"
    --pretrained
    --eval-batch-size "$eval_batch_size"
    --seq-length "$seq_length_eval"
    --finetune-mode "$finetune_mode"
    --mode evaluate
    --evaluation-metrics "$eval_metrics"
    --num-fewshot 0
    --load-from-checkpoint "$load_checkpoint"
    --project-name gateskip
    --limit 0
    "${additional_eval_args[@]}"
  )

  if [[ "$enable_logging" == true ]]; then
    train_args+=(--enable-logging)
  fi

  add_seed_args train_args
  add_seed_args eval_args

  run_experiment "${train_args[@]}"
  run_experiment "${eval_args[@]}"

  if [[ ${#additional_eval_invocations[@]} -gt 0 ]]; then
    local invocation
    for invocation in "${additional_eval_invocations[@]}"; do
      # shellcheck disable=SC2206
      local eval_variant_args=(${eval_args[@]} $invocation)
      run_experiment "${eval_variant_args[@]}"
    done
  fi
}
