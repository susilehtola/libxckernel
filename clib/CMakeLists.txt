# Driver for building the compiled libxckernel library as a generated
# artifact.  The repository ships only the generator and its tests; this
# file invokes the generator at configure time and then builds the
# emitted package.  The kernel selection is a configure-time choice:
#
#   cmake -S clib -B build \
#       -DXCKERNEL_FAMILIES=lda,gga,mgga_tau \
#       -DXCKERNEL_MAX_ORDER=3
#   cmake --build build
#
# Generation takes a small fraction of the time needed to compile the
# emitted code.  To produce a self-contained source tree for
# distribution (no Python needed downstream), run the generator
# directly:
#
#   python3 -m xckernel.catalog libxckernel <families> <order> c

cmake_minimum_required(VERSION 3.16)
project(libxckernel-driver LANGUAGES NONE)

set(XCKERNEL_FAMILIES "lda,gga,mgga_tau,mgga_lapl,mgga,cmgga_tau,hmgga"
    CACHE STRING "Comma-separated functional families to generate")
set(XCKERNEL_MAX_ORDER "4"
    CACHE STRING "Highest derivative order to generate (0-4)")

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(_gen_dir "${CMAKE_CURRENT_BINARY_DIR}/libxckernel")
set(_stamp "${_gen_dir}/.generated-for")
set(_want "${XCKERNEL_FAMILIES};${XCKERNEL_MAX_ORDER}")

set(_have "")
if(EXISTS "${_stamp}")
    file(READ "${_stamp}" _have)
endif()

if(NOT _have STREQUAL _want)
    message(STATUS "Generating libxckernel: families ${XCKERNEL_FAMILIES}, "
                   "order ${XCKERNEL_MAX_ORDER}")
    file(REMOVE_RECURSE "${_gen_dir}")
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m xckernel.catalog
                "${_gen_dir}" "${XCKERNEL_FAMILIES}"
                "${XCKERNEL_MAX_ORDER}" c
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
        RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "xckernel kernel generation failed")
    endif()
    file(WRITE "${_stamp}" "${_want}")
endif()

add_subdirectory("${_gen_dir}" "${CMAKE_CURRENT_BINARY_DIR}/pkg")
