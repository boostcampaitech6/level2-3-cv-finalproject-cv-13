clean:
	sudo docker rmi level2-3-cv-finalproject-cv-13-nginx:latest
	sudo docker rmi level2-3-cv-finalproject-cv-13-docker-fastapi:latest
	sudo docker buildx prune