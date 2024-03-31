cmake_minimum_required(VERSION 3.14)
project(day121)

set(SOURCE_FILES src/main.cpp extern/glad/src/gl.c)

add_executable(day121 ${SOURCE_FILES})

set_target_properties(day121 PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)

add_subdirectory(extern/glfw)
target_include_directories(day121 PRIVATE ${CMAKE_SOURCE_DIR} extern/glad/include)
target_link_libraries(day121 PRIVATE glfw)


link_directories(${CMAKE_SOURCE_DIR}/extern/freetype)
include_directories(${CMAKE_SOURCE_DIR}/extern/freetype/include)
target_link_libraries(day121 PRIVATE extern/freetype/freetype)


if(WIN32)
    target_link_libraries(day121 PRIVATE opengl32)
endif()
