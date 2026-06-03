# SENTINEL-Lite ROS 2 workspace. Run from ros2_ws/.
.PHONY: help build-drone build-computer build drone computer test clean

help:
	@echo "build-drone     colcon build cameras + msgs (Raspberry Pi)"
	@echo "build-computer  colcon build inference + msgs (Omen 15)"
	@echo "build           colcon build everything"
	@echo "drone           launch drone role (sources network + ws)"
	@echo "computer        launch computer role"
	@echo "test            run hardware/torch-free pytest (any machine)"
	@echo "clean           remove build/ install/ log/"

build-drone:
	colcon build --packages-up-to sentinel_camera

build-computer:
	colcon build --packages-up-to sentinel_inference

build:
	colcon build

drone:
	./scripts/run_drone.sh $(BACKEND)

computer:
	./scripts/run_computer.sh $(DEVICE) $(CKPT)

# Pure-logic tests: no ROS, no torch, no camera. Falls back to a manual runner
# when pytest isn't installed.
test:
	@python3 -m pytest src/sentinel_camera/test src/sentinel_inference/test test -q \
	  || python3 test/run_tests.py

clean:
	rm -rf build install log
