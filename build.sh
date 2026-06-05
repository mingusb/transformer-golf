#!/bin/bash
set -e

echo "=============================================="
echo "    MicroGPT Automatic Compilation Pipeline   "
echo "=============================================="

# Use the shared parent virtual environment
source /home/b/microgpt/venv/bin/activate

# --- Phase 1: Mathematical Synthesis ---
echo "[1/6] Running Z3 SAT/SMT Core Gate Synthesis..."
cd scripts
../venv/bin/python superoptimize_gates.py
cd ..

# --- Phase 2: Core Training & Ground Truth Generation ---
echo "[2/6] Running Model Training & Ground Truth Generation..."
./venv/bin/python compare_task.py > docs/last_run.log

# --- Phase 3: Hardware Transpilation ---
echo "[3/6] Transpiling PyTorch Graph into Discrete Hardware..."
./venv/bin/python scripts/transpile_circuit.py

# --- Phase 4: Universal Verification & Scaling Baselines ---
echo "[4/6] Verifying Hardware & Running Scaling Baselines..."
if [ ! -d "ngn-apl" ]; then
    echo "Cloning ngn/apl APL interpreter..."
    git clone https://github.com/abrudz/ngn-apl.git
fi
./venv/bin/python scripts/test_hardware_circuits.py >> docs/last_run.log
./venv/bin/python scripts/neurosymbolic_train.py >> docs/last_run.log
./venv/bin/python scripts/mlp_regex.py >> docs/last_run.log
echo "--- Testing Shannon's 1948 Markovian Text Generator ---" >> docs/last_run.log
./venv/bin/python scripts/shannon_markov.py --corpus regex_corpus.txt --order 5 --length 500 >> docs/last_run.log

# --- Phase 5: Mathematical Graph Optimization ---
echo "[5/6] Optimizing AST & Superoptimizing Graph..."
cd scripts
../venv/bin/python optimize.py
../venv/bin/python superoptimize.py >> ../docs/last_run.log
cd ..

# --- Phase 6: Visualization & Documentation ---
echo "[6/6] Generating Architecture Diagrams & Updating README..."
cd scripts
../venv/bin/python ast_to_latex_combined.py > combined.tex
pdflatex combined.tex > /dev/null 2>&1
pdfcrop combined.pdf combined_crop.pdf > /dev/null 2>&1
pdftocairo -png -r 300 combined_crop.pdf combined_crop
mv combined_crop-1.png ../docs/microgpt_architecture.png
rm combined*
cd ..

cat docs/last_run.log | ./venv/bin/python scripts/update_readme_logs.py

echo "[Done] Pipeline Complete."
echo "=============================================="
