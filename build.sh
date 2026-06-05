#!/bin/bash
set -e

echo "=============================================="
echo "    MicroGPT Automatic Compilation Pipeline   "
echo "=============================================="

# Use the shared parent virtual environment to prevent 20 subagents from exhausting the disk
source /home/b/microgpt/venv/bin/activate

# 0. Run the Task Evaluation and inject logs into README
echo "[1/4] Running Induction Head Demonstration & Updating README..."
./venv/bin/python scripts/update_readme_logic.py
./venv/bin/python compare_task.py > docs/last_run.log
if [ ! -d "ngn-apl" ]; then
    echo "Cloning ngn/apl APL interpreter..."
    git clone https://github.com/abrudz/ngn-apl.git
fi
./venv/bin/python scripts/test_hardware_circuits.py >> docs/last_run.log
./venv/bin/python scripts/neurosymbolic_train.py >> docs/last_run.log
./venv/bin/python scripts/mlp_regex.py >> docs/last_run.log
echo "--- Testing Shannon's 1948 Markovian Text Generator ---" >> docs/last_run.log
./venv/bin/python scripts/shannon_markov.py --corpus regex_corpus.txt --order 5 --length 500 >> docs/last_run.log
cat docs/last_run.log | ./venv/bin/python scripts/update_readme_logs.py

# 1. Run the Equality Saturation Compiler
echo "[2/4] Running Egglog Equality Saturation Optimizer..."
cd scripts
../venv/bin/python optimize.py
cd ..

# 1.5. Run SAT/SMT Superoptimization
echo "[2.5/4] Running Z3 SAT/SMT Superoptimizer..."
cd scripts
../venv/bin/python superoptimize.py
cd ..

# 2. Compile the Diagram
echo "[3/4] Generating Combined LaTeX AST Diagram..."
cd scripts
../venv/bin/python ast_to_latex_combined.py > combined.tex
pdflatex combined.tex > /dev/null 2>&1
pdfcrop combined.pdf combined_crop.pdf > /dev/null 2>&1
pdftocairo -png -r 300 combined_crop.pdf combined_crop
mv combined_crop-1.png ../docs/microgpt_architecture.png
rm combined*
cd ..

# 3. Finish
echo "[4/4] Done! Pipeline Complete."
echo "=============================================="
