# =============================================================================
# TIFF Enrichment Pipeline - Docker Management Makefile
# =============================================================================

# Load environment variables
include .env
export

# Default target
.DEFAULT_GOAL := help

# Docker Compose files
COMPOSE_FILE := docker-compose.yml
COMPOSE_PROJECT := tiff-pipeline

# Colors for output
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

# =============================================================================
# HELP
# =============================================================================

.PHONY: help
help: ## Show this help message
	@echo "$(CYAN)TIFF Enrichment Pipeline - Docker Management$(NC)"
	@echo ""
	@echo "$(YELLOW)Available commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Profiles:$(NC)"
	@echo "  $(GREEN)default$(NC)          - Main pipeline, database, and cache"
	@echo "  $(GREEN)admin$(NC)            - Add pgAdmin and Redis Commander"
	@echo "  $(GREEN)monitoring$(NC)       - Add Prometheus and Grafana"
	@echo "  $(GREEN)watcher$(NC)          - Add file watcher service"

# =============================================================================
# SETUP AND INITIALIZATION
# =============================================================================

.PHONY: init
init: ## Initialize the project (create directories, copy config files)
	@echo "$(CYAN)Initializing TIFF Pipeline project...$(NC)"
	@mkdir -p data/{input,output,failed} logs config/dev config/prod scripts monitoring/{prometheus,grafana/{dashboards,datasources}} src/{pipeline,tests}
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)Creating .env file from template...$(NC)"; \
		cp .env.template .env; \
		echo "$(RED)⚠️  Please edit .env file with your configuration!$(NC)"; \
	fi
	@if [ ! -f config/dev/settings.yml ]; then \
		echo "$(YELLOW)Creating development config...$(NC)"; \
		echo "environment: development" > config/dev/settings.yml; \
		echo "debug: true" >> config/dev/settings.yml; \
	fi
	@echo "$(GREEN)✓ Project initialized successfully!$(NC)"

.PHONY: check-env
check-env: ## Check if environment file exists and has required variables
	@if [ ! -f .env ]; then \
		echo "$(RED)Error: .env file not found!$(NC)"; \
		echo "Run 'make init' to create it from template"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ Environment file found$(NC)"

# =============================================================================
# DOCKER OPERATIONS
# =============================================================================

.PHONY: build
build: check-env ## Build all Docker images
	@echo "$(CYAN)Building Docker images...$(NC)"
	docker-compose -p $(COMPOSE_PROJECT) build --no-cache
	@echo "$(GREEN)✓ Build completed$(NC)"

.PHONY: build-quick
build-quick: check-env ## Build Docker images with cache
	@echo "$(CYAN)Building Docker images (with cache)...$(NC)"
	docker-compose -p $(COMPOSE_PROJECT) build
	@echo "$(GREEN)✓ Quick build completed$(NC)"

.PHONY: up
up: check-env ## Start the pipeline (core services only)
	@echo "$(CYAN)Starting TIFF Pipeline...$(NC)"
	docker-compose -p $(COMPOSE_PROJECT) up -d
	@echo "$(GREEN)✓ Pipeline started$(NC)"
	@echo "$(YELLOW)Check status with: make status$(NC)"

.PHONY: up-full
up-full: check-env ## Start pipeline with admin tools
	@echo "$(CYAN)Starting TIFF Pipeline with admin tools...$(NC)"
	docker-compose -p $(COMPOSE_PROJECT) --profile admin up -d
	@echo "$(GREEN)✓ Pipeline with admin tools started$(NC)"

.PHONY: up-monitoring
up-monitoring: check-env ## Start pipeline with monitoring stack
	@echo "$(CYAN)Starting TIFF Pipeline with monitoring...$(NC)"
	docker-compose -p $(COMPOSE_PROJECT) --profile monitoring up -d
	@echo "$(GREEN)✓ Pipeline with monitoring started$(NC)"

.PHONY: up-all
up-all: check-env ## Start all services (pipeline + admin + monitoring + watcher)
	@echo "$(CYAN)Starting all TIFF Pipeline services...$(NC)"
	docker-compose -p $(COMPOSE_PROJECT) --profile admin --profile monitoring --profile watcher up -d
	@echo "$(GREEN)✓ All services started$(NC)"

.PHONY: down
down: ## Stop and remove all containers
	@echo "$(CYAN)Stopping TIFF Pipeline...$(NC)"
	docker-compose -p $(COMPOSE_PROJECT) --profile admin --profile monitoring --profile watcher down
	@echo "$(GREEN)✓ Pipeline stopped$(NC)"

.PHONY: restart
restart: down up ## Restart the pipeline
	@echo "$(GREEN)✓ Pipeline restarted$(NC)"

# =============================================================================
# STATUS AND MONITORING
# =============================================================================

.PHONY: status
status: ## Show status of all containers
	@echo "$(CYAN)TIFF Pipeline Status:$(NC)"
	@docker-compose -p $(COMPOSE_PROJECT) ps

.PHONY: logs
logs: ## Show logs for all services
	docker-compose -p $(COMPOSE_PROJECT) logs -f

.PHONY: logs-pipeline
logs-pipeline: ## Show logs for the main pipeline service
	docker-compose -p $(COMPOSE_PROJECT) logs -f tiff-pipeline

.PHONY: logs-db
logs-db: ## Show database logs
	docker-compose -p $(COMPOSE_PROJECT) logs -f postgres

.PHONY: health
health: ## Check health of all services
	@echo "$(CYAN)Service Health Check:$(NC)"
	@for service in $$(docker-compose -p $(COMPOSE_PROJECT) config --services); do \
		status=$$(docker-compose -p $(COMPOSE_PROJECT) ps -q $$service | xargs docker inspect --format='{{.State.Health.Status}}' 2>/dev/null || echo "no-healthcheck"); \
		if [ "$$status" = "healthy" ]; then \
			echo "  $(GREEN)✓ $$service: healthy$(NC)"; \
		elif [ "$$status" = "unhealthy" ]; then \
			echo "  $(RED)✗ $$service: unhealthy$(NC)"; \
		elif [ "$$status" = "starting" ]; then \
			echo "  $(YELLOW)⏳ $$service: starting$(NC)"; \
		else \
			echo "  $(YELLOW)? $$service: no health check$(NC)"; \
		fi; \
	done

# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

.PHONY: db-shell
db-shell: ## Connect to PostgreSQL database shell
	docker-compose -p $(COMPOSE_PROJECT) exec postgres psql -U pipeline -d tiff_pipeline

.PHONY: db-backup
db-backup: ## Create database backup
	@mkdir -p backups
	@backup_file="backups/tiff_pipeline_$$(date +%Y%m%d_%H%M%S).sql"; \
	echo "$(CYAN)Creating database backup: $$backup_file$(NC)"; \
	docker-compose -p $(COMPOSE_PROJECT) exec -T postgres pg_dump -U pipeline -d tiff_pipeline > $$backup_file; \
	echo "$(GREEN)✓ Backup created: $$backup_file$(NC)"

.PHONY: db-restore
db-restore: ## Restore database from backup (usage: make db-restore BACKUP_FILE=path/to/backup.sql)
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "$(RED)Error: Please specify BACKUP_FILE=path/to/backup.sql$(NC)"; \
		exit 1; \
	fi
	@echo "$(CYAN)Restoring database from: $(BACKUP_FILE)$(NC)"
	docker-compose -p $(COMPOSE_PROJECT) exec -T postgres psql -U pipeline -d tiff_pipeline < $(BACKUP_FILE)
	@echo "$(GREEN)✓ Database restored$(NC)"

.PHONY: db-reset
db-reset: ## Reset database (WARNING: This will delete all data!)
	@echo "$(RED)⚠️  WARNING: This will delete ALL data in the database!$(NC)"
	@read -p "Are you sure? Type 'yes' to continue: " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		echo "$(CYAN)Resetting database...$(NC)"; \
		docker-compose -p $(COMPOSE_PROJECT) down postgres; \
		docker volume rm $(COMPOSE_PROJECT)_postgres_data 2>/dev/null || true; \
		docker-compose -p $(COMPOSE_PROJECT) up -d postgres; \
		echo "$(GREEN)✓ Database reset completed$(NC)"; \
	else \
		echo "$(YELLOW)Database reset cancelled$(NC)"; \
	fi

# =============================================================================
# DEVELOPMENT AND TESTING
# =============================================================================

.PHONY: test
test: ## Run tests in the pipeline container
	@echo "$(CYAN)Running tests...$(NC)"
	docker-compose -p $(COMPOSE_PROJECT) exec tiff-pipeline python -m pytest tests/ -v --cov=src/
	@echo "$(GREEN)✓ Tests completed$(NC)"

.PHONY: test-build
test-build: ## Build and run tests
	@echo "$(CYAN)Building and running tests...$(NC)"
	docker-compose -p $(COMPOSE_PROJECT) build tiff-pipeline
	docker-compose -p $(COMPOSE_PROJECT) run --rm tiff-pipeline python -m pytest tests/ -v --cov=src/

.PHONY: lint
lint: ## Run code linting
	@echo "$(CYAN)Running linting...$(NC)"
	docker-compose -p $(COMPOSE_PROJECT) exec tiff-pipeline python -m flake8 src/
	docker-compose -p $(COMPOSE_PROJECT) exec tiff-pipeline python -m black --check src/
	@echo "$(GREEN)✓ Linting completed$(NC)"

.PHONY: format
format: ## Format code
	@echo "$(CYAN)Formatting code...$(NC)"
	docker-compose -p $(COMPOSE_PROJECT) exec tiff-pipeline python -m black src/
	docker-compose -p $(COMPOSE_PROJECT) exec tiff-pipeline python -m isort src/
	@echo "$(GREEN)✓ Code formatted$(NC)"

.PHONY: shell
shell: ## Open shell in the pipeline container
	docker-compose -p $(COMPOSE_PROJECT) exec tiff-pipeline bash

# =============================================================================
# PIPELINE OPERATIONS
# =============================================================================

.PHONY: process-sample
process-sample: ## Process sample TIFF files (if available)
	@if [ ! -d "data/input" ] || [ -z "$$(ls -A data/input)" ]; then \
		echo "$(YELLOW)No files found in data/input directory$(NC)"; \
		echo "Please add some TIFF files to process"; \
	else \
		echo "$(CYAN)Processing files in data/input...$(NC)"; \
		docker-compose -p $(COMPOSE_PROJECT) exec tiff-pipeline python -m pipeline.main; \
		echo "$(GREEN)✓ Processing completed$(NC)"; \
	fi

.PHONY: watch-start
watch-start: ## Start file watcher service
	@echo "$(CYAN)Starting file watcher...$(NC)"
	docker-compose -p $(COMPOSE_PROJECT) --profile watcher up -d file-watcher
	@echo "$(GREEN)✓ File watcher started$(NC)"

.PHONY: watch-stop
watch-stop: ## Stop file watcher service
	@echo "$(CYAN)Stopping file watcher...$(NC)"
	docker-compose -p $(COMPOSE_PROJECT) stop file-watcher
	@echo "$(GREEN)✓ File watcher stopped$(NC)"

# =============================================================================
# CLEANUP
# =============================================================================

.PHONY: clean
clean: ## Remove containers and volumes
	@echo "$(CYAN)Cleaning up containers and volumes...$(NC)"
	docker-compose -p $(COMPOSE_PROJECT) --profile admin --profile monitoring --profile watcher down -v
	@echo "$(GREEN)✓ Cleanup completed$(NC)"

.PHONY: clean-all
clean-all: clean ## Remove everything including images
	@echo "$(CYAN)Removing Docker images...$(NC)"
	docker-compose -p $(COMPOSE_PROJECT) down --rmi all
	@echo "$(GREEN)✓ Complete cleanup finished$(NC)"

.PHONY: clean-logs
clean-logs: ## Clear log files
	@echo "$(CYAN)Clearing log files...$(NC)"
	rm -rf logs/*
	mkdir -p logs
	@echo "$(GREEN)✓ Log files cleared$(NC)"

# =============================================================================
# MONITORING SHORTCUTS
# =============================================================================

.PHONY: pgadmin
pgadmin: ## Open pgAdmin URL
	@echo "$(CYAN)pgAdmin URL: http://localhost:$(PGADMIN_PORT)$(NC)"
	@echo "Email: $(PGADMIN_EMAIL)"
	@echo "Password: [check your .env file]"

.PHONY: grafana
grafana: ## Open Grafana URL
	@echo "$(CYAN)Grafana URL: http://localhost:$(GRAFANA_PORT)$(NC)"
	@echo "Username: admin"
	@echo "Password: [check your .env file]"

.PHONY: prometheus
prometheus: ## Open Prometheus URL
	@echo "$(CYAN)Prometheus URL: http://localhost:$(PROMETHEUS_PORT)$(NC)"

# =============================================================================
# PRODUCTION DEPLOYMENT
# =============================================================================

.PHONY: deploy-prod
deploy-prod: ## Deploy to production (requires additional setup)
	@echo "$(RED)⚠️  Production deployment requires additional configuration!$(NC)"
	@echo "Please ensure you have:"
	@echo "1. Configured production environment variables"
	@echo "2. Set up SSL certificates"
	@echo "3. Configured backup strategies"
	@echo "4. Set up monitoring and alerting"
	@read -p "Continue with production deployment? (yes/no): " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		ENVIRONMENT=production docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d; \
		echo "$(GREEN)✓ Production deployment started$(NC)"; \
	else \
		echo "$(YELLOW)Production deployment cancelled$(NC)"; \
	fi

# =============================================================================
# PRODUCTION OPERATIONS
# =============================================================================

.PHONY: up-prod
up-prod: check-env ## Start all containers in production mode
	@echo "$(CYAN)Starting TIFF Pipeline (production)...$(NC)"
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
	@echo "$(GREEN)✓ Production pipeline started$(NC)"

.PHONY: down-prod
down-prod: ## Stop all containers in production mode
	@echo "$(CYAN)Stopping TIFF Pipeline (production)...$(NC)"
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml down
	@echo "$(GREEN)✓ Production pipeline stopped$(NC)"

.PHONY: status-prod
status-prod: ## Show status of all containers in production mode
	@echo "$(CYAN)TIFF Pipeline Status (production):$(NC)"
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml ps

# =============================================================================
# UTILITIES
# =============================================================================

.PHONY: update-deps
update-deps: ## Update Python dependencies
	@echo "$(CYAN)Updating Python dependencies...$(NC)"
	docker-compose -p $(COMPOSE_PROJECT) run --rm tiff-pipeline pip list --outdated
	@echo "$(YELLOW)To update, modify requirements.txt and rebuild$(NC)"

.PHONY: version
version: ## Show version information
	@echo "$(CYAN)TIFF Enrichment Pipeline$(NC)"
	@echo "Docker Compose version: $$(docker-compose version --short)"
	@echo "Docker version: $$(docker version --format '{{.Client.Version}}')"
	@if [ -f "VERSION" ]; then \
		echo "Pipeline version: $$(cat VERSION)"; \
	fi