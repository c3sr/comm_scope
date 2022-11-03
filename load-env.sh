echo_and_do() {
    echo "$@"
    $@
}

if [[ ${LMOD_SYSTEM_NAME} =~ crusher ]]; then
    echo "LMOD_SYSTEM_NAME matched crusher"
    echo_and_do module load cmake
    echo_and_do module load rocm/5.3.0
else
    echo "UNRECOGNIZED HOST $(hostname)"
fi