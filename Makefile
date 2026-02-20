init:
	-docker rmi slt-renewal-model
	docker build -t slt-renewal-model .
	cp -n slt_season_renewal.env.dist slt_season_renewal.env | echo 'slt_season_renewal.env is initialized!'

features:
	COMPOSE_USER=`whoami` docker-compose up -d prometheus cadvisor
	COMPOSE_USER=`whoami` docker-compose up features
	docker-compose logs -f features

train:
	COMPOSE_USER=`whoami` docker-compose up -d prometheus cadvisor
	COMPOSE_USER=`whoami` docker-compose up train
	docker-compose logs -f train

score:
	COMPOSE_USER=`whoami` docker-compose up score
	docker-compose logs -f score

clean:
	docker-compose kill features
	docker-compose rm -f features
	docker-compose kill train
	docker-compose rm -f train
	docker-compose kill score
	docker-compose rm -f score
	docker-compose stop prometheus cadvisor

test:
	docker build -t slt-renewal-model .
	docker run -it --rm --env-file slt_season_renewal.env slt-renewal-model features
	docker run -it --rm --env-file slt_season_renewal.env slt-renewal-model train
	docker run -it --rm --env-file slt_season_renewal.env slt-renewal-model score
	docker system prune -f