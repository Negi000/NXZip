cmake_minimum_required(VERSION 3.15)
project(NexusImageCompressor VERSION 1.0.0 LANGUAGES CXX)

# C++17 標準を使用
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# コンパイラ最適化設定
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        add_compile_options(/O2 /Oi /Ot /GL)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG")
    else()
        add_compile_options(-O3 -march=native -mtune=native -ffast-math)
    endif()
endif()

# SIMD サポート
if(MSVC)
    add_compile_options(/arch:AVX2)
else()
    add_compile_options(-mavx2 -msse4.1)
endif()

# ソースファイル
set(SOURCES
    src/nexus_image_compressor.cpp
    src/main.cpp
)

# ヘッダーファイル
set(HEADERS
    src/nexus_image_compressor.hpp
)

# 実行ファイル作成
add_executable(nexus_compressor ${SOURCES} ${HEADERS})

# インクルードディレクトリ
target_include_directories(nexus_compressor PRIVATE src)

# Windows固有設定
if(WIN32)
    target_compile_definitions(nexus_compressor PRIVATE 
        _CRT_SECURE_NO_WARNINGS
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
endif()

# デバッグ情報
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(nexus_compressor PRIVATE DEBUG)
endif()

# インストール設定
install(TARGETS nexus_compressor
    RUNTIME DESTINATION bin
)

# テスト用のカスタムターゲット
add_custom_target(test_run
    COMMAND nexus_compressor test
    DEPENDS nexus_compressor
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running NEXUS Image Compressor tests"
)

# パフォーマンステスト用
add_custom_target(benchmark
    COMMAND nexus_compressor compress ${CMAKE_SOURCE_DIR}/test-data/sample.png ultra_fast
    COMMAND nexus_compressor compress ${CMAKE_SOURCE_DIR}/test-data/sample.png balanced  
    COMMAND nexus_compressor compress ${CMAKE_SOURCE_DIR}/test-data/sample.png maximum
    DEPENDS nexus_compressor
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running performance benchmarks"
)

# コンパイル情報表示
message(STATUS "🚀 NEXUS Image Compressor C++ Configuration")
message(STATUS "   Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "   Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "   C++ Standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "   Platform: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "   Optimizations: Enabled (O3/O2 + SIMD)")
else()
    message(STATUS "   Optimizations: Disabled (Debug Mode)")
endif()
