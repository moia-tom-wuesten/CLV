#!/bin/bash
VENVS=( "pytorch2.0" "pytorch1.13.x" "tensorflow2.11" )
#VENVS=( "tensorflow2.11" )
DATASETS=( "bank" "moia" "cdnow" "grocery" "retail") 
#DATASETS=( "cdnow" "grocery" "retail") 
BATCHSSIZE=( 32 64 128)
run_benchmark(){
    for batch_size in "${BATCHSSIZE[@]}"
    do
        echo $batch_size
        #poetry run python single_benchmark.py --file results.csv --framework $3 --batch_size $batch_size --mode train --device cpu --dataset $2
        poetry run python single_benchmark.py --file results_mac_interference_cpu.csv --framework $3 --batch_size $batch_size --mode inference_cpu --device cpu --dataset $2
        #poetry run python single_benchmark.py --file results_mac_interference_gpu.csv --framework $3 --batch_size $batch_size --mode inference_gpu --device mps --dataset $2
    done
    #poetry run python benchmark.py --device "$1" --dataset "$2"
    #single_benchmark.py --file results.csv --framework pytorch2 --batch_size 64 --mode train --device cpu --dataset bank 
    echo "$3"
}
switch_venv(){
    cd $1
    echo `pwd`
    #poetry env use ~/.pyenv/versions/3.10.6/bin/python
    source .venv/bin/activate
}
echo `pwd`
cd src/CLV/benchmarking/
echo `pwd`
for venv in "${VENVS[@]}"
do
    switch_venv $venv
    cd ..
    echo `pwd`
    for dataset in "${DATASETS[@]}"
    do
        if [ "$venv" = "pytorch1.13.x" ]; then
            run_benchmark "cpu" "$dataset" "pytorch1"
        elif [ "$venv" = "pytorch2.0" ]; then
            run_benchmark "cpu" "$dataset" "pytorch2"
        elif [ "$venv" = "tensorflow2.11" ]; then
            run_benchmark "cpu" "$dataset" "tf2"
        else
            echo "Wrong venv version"
        fi
        
        
    done
done
