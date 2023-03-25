#!/bin/bash
DIR=".venv/"
VENVS=( "pytorch1.13.x" "pytorch2.0" )
install_dependencies() {
    if test -d "$DIR"; then
    echo "$DIR exists."
    poetry env use ~/.pyenv/versions/3.10.6/bin/python
    source .venv/bin/activate
    poetry install
    deactivate
    fi

}

cd src/CLV/benchmarking/
## now loop through the above List
for venv in "${VENVS[@]}"
do
    cd "$venv"
    install_dependencies
    echo `pwd`
    cd ..
    echo `pwd`
   # or do whatever with individual element of the List
done