cd $HOME/GateSkip

. jobs/environment.sh

"${APPTAINER_RUN[@]}" python -m pip install plotly

"${APPTAINER_RUN[@]}" python collect_results.py $WORK/GateSkip/cache/results/GateSkip-frozen-backbone-MLP-shared_cot_results.json
