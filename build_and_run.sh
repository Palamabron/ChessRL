#!/bin/bash
# build_and_run.sh - Build and run script for Chess RL

# Create Dockerfiles from the artifact contents
cp Dockerfile.train Dockerfile.train
cp Dockerfile.gui Dockerfile.gui 
cp docker-compose.yml docker-compose.yml

# Function to check if NVIDIA GPU is available
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected. Setting up GPU support."
        return 0
    else
        echo "No NVIDIA GPU detected. Using CPU-only mode."
        return 1
    fi
}

# Build the Docker images
build_images() {
    echo "Building Docker images..."
    docker-compose build
}

# Run the training container
run_training() {
    echo "Starting training container..."
    if check_gpu; then
        docker-compose up train
    else
        # Remove GPU-specific configuration for CPU-only systems
        sed -i 's/deploy:.*$//' docker-compose.yml
        sed -i 's/resources:.*$//' docker-compose.yml
        sed -i 's/reservations:.*$//' docker-compose.yml
        sed -i 's/devices:.*$//' docker-compose.yml
        sed -i 's/- driver: nvidia.*$//' docker-compose.yml
        sed -i 's/count: 1.*$//' docker-compose.yml
        sed -i 's/capabilities: \[gpu\].*$//' docker-compose.yml
        sed -i 's/CUDA_VISIBLE_DEVICES=0//' docker-compose.yml
        docker-compose up train
    fi
}

# Run the GUI container
run_gui() {
    echo "Starting GUI container..."
    # Set up X11 forwarding
    xhost +local:docker || true
    docker-compose up play
}

# Main execution
case "$1" in
    "build")
        build_images
        ;;
    "train")
        run_training
        ;;
    "play")
        run_gui
        ;;
    *)
        echo "Usage: $0 {build|train|play}"
        echo "  build: Build the Docker images"
        echo "  train: Run the training container"
        echo "  play: Run the GUI container"
        exit 1
        ;;
esac

exit 0