REPO_PATH=/home/user/text-mol
CACHE_PATH=/home/user/.cache
DATA_PATH=/data/text-mol
IMAGE_NAME_TAG=user/mol-llm:v1

build-image:
	docker build -t $(IMAGE_NAME_TAG) .

init-container:
	docker run -it \
	--gpus all \
	-v $(REPO_PATH):/text-mol \
	-v $(DATA_PATH):/data \
	-v $(CACHE_PATH):/root/.cache/ \
	--shm-size=32g \
	--name mol-llm \
	$(IMAGE_NAME_TAG) \
	/bin/bash