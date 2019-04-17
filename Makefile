.PHONY: build jupyter api bash

build:
	docker-compose build

jupyter:
	docker-compose run --service-ports --rm analytics \
		jupyter lab --ip=0.0.0.0 --allow-root --LabApp.token=''

api:
	docker-compose up -d analytics

bash:
	docker-compose run --rm analytics bash
