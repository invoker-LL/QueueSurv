#!/bin/bash

gpuid=$1
config=$2

### Dataset Information
declare -a dataroots=(
	'path/to/tcga_kirc'
)


task='COADREAD_survival'
target_col='dss_survival_days'
split_names='train,test'

bash "./scripts/survival/${config}.sh" 0 $task $target_col 'survival/TCGA_KIRC_overall_survival_k=0' $split_names "${dataroots[@]}" &
bash "./scripts/survival/${config}.sh" 1 $task $target_col 'survival/TCGA_KIRC_overall_survival_k=1' $split_names "${dataroots[@]}" &
bash "./scripts/survival/${config}.sh" 2 $task $target_col 'survival/TCGA_KIRC_overall_survival_k=2' $split_names "${dataroots[@]}"
bash "./scripts/survival/${config}.sh" 1 $task $target_col 'survival/TCGA_KIRC_overall_survival_k=3' $split_names "${dataroots[@]}" &
bash "./scripts/survival/${config}.sh" 2 $task $target_col 'survival/TCGA_KIRC_overall_survival_k=4' $split_names "${dataroots[@]}"