file(GLOB PYTHON_FILES_TO_COPY "${CMAKE_CURRENT_LIST_DIR}/python/*.py")
file(COPY ${PYTHON_FILES_TO_COPY} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/qupled)
file(GLOB PYTHON_FILES_TO_COPY "${CMAKE_CURRENT_LIST_DIR}/python/__init.py__")
file(COPY ${PYTHON_FILES_TO_COPY} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/qupled)