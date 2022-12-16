echo_and_do() {
    echo "$@"
    $@
}

if [[ ${LMOD_SYSTEM_NAME} =~ crusher ]]; then
    echo "LMOD_SYSTEM_NAME matched crusher"
    echo_and_do module load cmake
    echo_and_do module load rocm/5.4.0
elif [[ ${HOSTNAME} =~ caraway ]]; then
    echo "HOSTNAME matched caraway"
    echo_and_do module load cmake
    echo_and_do module load rocm/5.2.0
    echo_and_do module load numa/2.0.11
elif [[ `hostname` =~ vortex ]]; then
    echo "hostname matched vortex"
    echo_and_do module load cmake/3.23.1
    echo_and_do module load gcc/8.3.1
    echo_and_do module load cuda/11.7.0
else
    echo "UNRECOGNIZED HOST $(hostname)"
fi