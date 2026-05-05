.PHONY: bootstrap up down logs test eval extract clean

# first time setup - builds images, starts everything, copies env file
bootstrap:
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env - add your GEMINI_API_KEY before running"; fi
	docker compose build --no-cache
	docker compose up -d
	@echo ""
	@echo "  UI:      http://localhost:3000"
	@echo "  API:     http://localhost:8000"
	@echo "  Health:  http://localhost:8000/api/health"

# start (builds only if images are missing or changed)
up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

# runs pytest inside the backend container
test:
	docker compose exec backend pytest tests/ -v

# runs the scoring script against the live API
# needs pip install requests jsonschema on your host machine
eval:
	pip install -q requests jsonschema
	python eval/score.py --api-url http://localhost:8000

# submit a job from the terminal
# usage: make extract LAT="48.8566" LON="2.3522" RADIUS_KM="50"
LAT ?= 48.8566
LON ?= 2.3522
RADIUS_KM ?= 50

extract:
	@echo "Submitting job: lat=$(LAT), lon=$(LON), radius=$(RADIUS_KM)km"
	@curl -s -X POST http://localhost:8000/api/jobs \
	  -H "Content-Type: application/json" \
	  -d "{\"latitude\": $(LAT), \"longitude\": $(LON), \"radius_km\": $(RADIUS_KM)}" \
	  | python3 -m json.tool

# tears everything down including volumes and built images
clean:
	docker compose down -v --rmi local
