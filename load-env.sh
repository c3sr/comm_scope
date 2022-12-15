echo_and_do() {
    echo "$@"
    $@
}

if [[ ${LMOD_SYSTEM_NAME} =~ crusher ]]; then
    echo "LMOD_SYSTEM_NAME matched crusher"
    echo_and_do module load cmake
    echo_and_do module load rocm/5.4.0
elif [[ $(hostname) =~ caraway ]]; then
    echo "LMOD_SYSTEM_NAME matched caraway"
    echo_and_do module load cmake
    echo_and_do module load rocm/5.2.0
    echo_and_do module load numa/2.0.11
else
    echo "UNRECOGNIZED HOST $(hostname)"
fi