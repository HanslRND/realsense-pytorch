cmake_minimum_required(VERSION 3.8)
project(realsense_yolov7)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(realsense_yolov7)
ament_python_install_package(models)
ament_python_install_package(utils)

install(PROGRAMS
  main.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME realsense_yolov7_node
)

install(PROGRAMS
  export_tensorrt.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME export_tensorrt
)

install(DIRECTORY
  models/weight
  DESTINATION share/${PROJECT_NAME}/models
)

ament_package()
