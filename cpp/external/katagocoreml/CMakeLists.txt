# katagocoreml - KataGo to Core ML Converter (vendored)
# Simplified build for use as a subdirectory of KataGo.
#
# Note: We deliberately avoid linking against CMake's abseil/protobuf targets
# (find_package(absl)) because their INTERFACE_LINK_LIBRARIES propagate
# "-Wl,-framework,CoreFoundation" to the final executable, which swiftc
# (used as the linker for the katago Swift/C++ hybrid) does not understand.
# Instead, we use pkg-config for include/link flags, which produces
# swiftc-compatible flags like "-framework CoreFoundation".

# ============================================================================
# External Dependencies
# ============================================================================

find_package(ZLIB REQUIRED)
find_package(Protobuf REQUIRED)  # Needed for protoc executable and include dirs
find_package(PkgConfig REQUIRED)
pkg_check_modules(KATAGOCOREML_ABSEIL REQUIRED
    absl_base absl_log absl_log_internal_check_op absl_log_internal_message
    absl_hash absl_strings absl_status absl_statusor
)

# Export link flags to parent scope for the final executable
pkg_check_modules(KATAGOCOREML_ALL_DEPS REQUIRED
    protobuf
    absl_base absl_log absl_log_internal_check_op absl_log_internal_message
    absl_hash absl_strings absl_status absl_statusor
)
set(KATAGOCOREML_DEP_LDFLAGS ${KATAGOCOREML_ALL_DEPS_LDFLAGS} PARENT_SCOPE)

# ============================================================================
# Proto Files (compile from source)
# ============================================================================

set(COREMLTOOLS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/vendor")
set(PROTO_DIR "${COREMLTOOLS_ROOT}/mlmodel/format")
set(PROTO_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/proto")
file(MAKE_DIRECTORY ${PROTO_GENERATED_DIR})

# Get all proto files
file(GLOB PROTO_FILES "${PROTO_DIR}/*.proto")

# Generate C++ from all proto files
set(PROTO_SRCS)
set(PROTO_HDRS)

foreach(PROTO_FILE ${PROTO_FILES})
    get_filename_component(PROTO_NAME ${PROTO_FILE} NAME_WE)
    set(PROTO_SRC "${PROTO_GENERATED_DIR}/${PROTO_NAME}.pb.cc")
    set(PROTO_HDR "${PROTO_GENERATED_DIR}/${PROTO_NAME}.pb.h")
    list(APPEND PROTO_SRCS ${PROTO_SRC})
    list(APPEND PROTO_HDRS ${PROTO_HDR})

    add_custom_command(
        OUTPUT ${PROTO_SRC} ${PROTO_HDR}
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
        ARGS --cpp_out=${PROTO_GENERATED_DIR}
             -I${PROTO_DIR}
             ${PROTO_FILE}
        DEPENDS ${PROTO_FILE}
        COMMENT "Generating C++ from ${PROTO_NAME}.proto"
        VERBATIM
    )
endforeach()

# ============================================================================
# MILBlob Sources (vendored from coremltools)
# ============================================================================

set(MILBLOB_DIR "${COREMLTOOLS_ROOT}/mlmodel/src/MILBlob")
set(MILBLOB_SRCS
    "${MILBLOB_DIR}/Blob/FileWriter.cpp"
    "${MILBLOB_DIR}/Blob/StorageWriter.cpp"
    "${MILBLOB_DIR}/Blob/StorageReader.cpp"
    "${MILBLOB_DIR}/Blob/MMapFileReader.cpp"
    "${MILBLOB_DIR}/Blob/MMapFileReaderFactory.cpp"
    "${MILBLOB_DIR}/SubByteTypes.cpp"
    "${MILBLOB_DIR}/Fp8.cpp"
    "${MILBLOB_DIR}/Fp16.cpp"
)

# ============================================================================
# ModelPackage Sources (vendored from coremltools)
# ============================================================================

set(MODELPACKAGE_DIR "${COREMLTOOLS_ROOT}/modelpackage/src")
set(MODELPACKAGE_SRCS
    "${MODELPACKAGE_DIR}/ModelPackage.cpp"
    "${MODELPACKAGE_DIR}/utils/JsonMap.cpp"
)

# ============================================================================
# KataGoCoreML Library Sources
# ============================================================================

set(KATAGOCOREML_SRCS
    src/parser/KataGoParser.cpp
    src/builder/MILBuilder.cpp
    src/builder/Operations.cpp
    src/serializer/CoreMLSerializer.cpp
    src/serializer/WeightSerializer.cpp
    src/Converter.cpp
)

# ============================================================================
# Library Target
# ============================================================================

add_library(katagocoreml STATIC
    ${KATAGOCOREML_SRCS}
    ${PROTO_SRCS}
    ${MILBLOB_SRCS}
    ${MODELPACKAGE_SRCS}
)

target_include_directories(katagocoreml
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${PROTO_GENERATED_DIR}
        ${MILBLOB_DIR}/..
        ${MODELPACKAGE_DIR}
        ${COREMLTOOLS_ROOT}/deps/nlohmann
        ${COREMLTOOLS_ROOT}/deps/FP16/include
        ${Protobuf_INCLUDE_DIRS}
        ${KATAGOCOREML_ABSEIL_INCLUDE_DIRS}
)

# Only link ZLIB as a CMake target (no swiftc-incompatible flags).
# Protobuf and abseil are linked via pkg-config LDFLAGS in the parent.
target_link_libraries(katagocoreml PRIVATE ZLIB::ZLIB)

target_compile_definitions(katagocoreml PRIVATE APPLE_BUILD=1)
