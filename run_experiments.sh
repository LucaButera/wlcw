#!/usr/bin/env bash
set -euo pipefail

# Experiment runner for "Why Do Time Series Models Need Long Context Windows?"
# Usage:
#   ./run_experiments.sh [figure2|figure3|figure4|figure5-6|figure7-8] [extra hydra arguments...]

PYTHON="${PYTHON:-python}"

run_config() {
    local script="$1"
    local config_file="$2"
    shift 2
    local extra_args=("$@")

    local config_dir
    config_dir="$(cd "$(dirname "$config_file")" && pwd)"
    local config_name
    config_name="$(basename "$config_file" .yaml)"

    echo "Running: $script | Config: $config_name"
    "$PYTHON" "$script" --config-dir "$config_dir" -cn="$config_name" ${extra_args[@]+"${extra_args[@]}"}
}

run_figure2() {
    echo ">>> Running Figure 2..."
    for cfg in reference_configs/figure2/*.yaml; do
        run_config test_pretrained.py "$cfg" "$@"
    done
}

run_figure3() {
    echo ">>> Running Figure 3..."
    for cfg in reference_configs/figure3/*.yaml; do
        run_config main.py "$cfg" "$@"
    done
}

run_figure4() {
    echo ">>> Running Figure 4..."
    for cfg in reference_configs/figure4/*.yaml; do
        run_config main.py "$cfg" "$@"
    done
}

run_figure5_6() {
    echo ">>> Running Figures 5-6..."
    for cfg in reference_configs/figure5-6/*_standard.yaml; do
        run_config test_pretrained.py "$cfg" "$@"
    done
    for cfg in reference_configs/figure5-6/*_multidomain.yaml reference_configs/figure5-6/*_singledomain.yaml; do
        run_config main.py "$cfg" "$@"
    done
}

run_figure7_8() {
    echo ">>> Running Figures 7-8..."
    for cfg in reference_configs/figure7-8/*.yaml; do
        run_config main.py "$cfg" "$@"
    done
}

show_help() {
    echo "Usage: $0 [TARGET] [EXTRA_HYDRA_ARGS...]"
    echo ""
    echo "Targets:"
    echo "  figure2         Results for Figure 2"
    echo "  figure3         Results for Figure 3"
    echo "  figure4         Results for Figure 4"
    echo "  figure5-6       Results for Figures 5 & 6"
    echo "  figure7-8       Results for Figures 7 & 8"
    echo ""
    echo "Examples:"
    echo "  $0 figure2"
    echo "  $0 figure3 trainer.max_epochs=10 seed=42"
    echo "  $0 figure5-6"
    echo "  $0 figure7-8 hydra.sweeper.params.seed=1"
}

if [ $# -lt 1 ]; then
    show_help
    exit 1
fi

TARGET="$1"
shift

case "$TARGET" in
    figure2)
        run_figure2 "$@"
        ;;
    figure3)
        run_figure3 "$@"
        ;;
    figure4)
        run_figure4 "$@"
        ;;
    figure5-6)
        run_figure5_6 "$@"
        ;;
    figure7-8)
        run_figure7_8 "$@"
        ;;
    -h|--help|help)
        show_help
        exit 0
        ;;
    *)
        echo "Error: Unknown target '$TARGET'"
        echo ""
        show_help
        exit 1
        ;;
esac
