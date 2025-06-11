# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Prerequisites
- Docker 20.10+ installed
- Docker Compose 2.0+ installed
- 4GB+ RAM available
- 10GB+ disk space

### 1. Clone the Repository
```bash
git clone https://github.com/casper-aurai/tiff-enrichment-pipeline.git
cd tiff-enrichment-pipeline
```

### 2. Initialize the Project
```bash
make init
```
This creates directory structure and copies `.env.template` to `.env`.

### 3. Configure Environment
Edit the `.env` file with your settings:
```bash
vim .env
```

**Required settings:**
```bash
POSTGRES_PASSWORD=your_secure_password
USGS_API_KEY=your_usgs_api_key    # Optional
SLACK_WEBHOOK_URL=your_webhook    # Optional
```

### 4. Start the Pipeline
```bash
# Basic setup (pipeline + database + cache)
make up

# Or with admin tools (pgAdmin, Redis Commander)
make up-full

# Or with everything (admin + monitoring + file watcher)
make up-all
```

### 5. Verify Installation
```bash
# Check service status
make status

# Check health
make health

# View logs
make logs
```

### 6. Process TIFF Files
```bash
# Add your TIFF files to data/input/
cp /path/to/your/files/*.tiff data/input/

# Process them
make process-sample
```

### 7. View Results
- **Processed files**: `data/output/tiff/`
- **Metadata**: `data/output/json/`
- **Database**: `make db-shell`

## 🎯 Common Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make up` | Start basic pipeline |
| `make down` | Stop all services |
| `make logs` | View all logs |
| `make shell` | Open container shell |
| `make test` | Run tests |
| `make clean` | Clean up containers |

## 🔧 Admin Interfaces

With `make up-full` or `make up-all`:

- **pgAdmin**: http://localhost:8080
- **Redis Commander**: http://localhost:8081
- **Grafana**: http://localhost:3000 (with monitoring)
- **Prometheus**: http://localhost:9090 (with monitoring)

## 🐛 Troubleshooting

### Services won't start
```bash
make logs
docker system prune -f
make build
```

### Database issues
```bash
make db-reset  # ⚠️ Deletes all data
make db-backup
```

### Port conflicts
Edit `.env` file and change port numbers:
```bash
POSTGRES_PORT=5433
PGADMIN_PORT=8081
```

### Out of disk space
```bash
make clean
docker system prune -a
```

## 📊 Production Deployment

For production use:
```bash
make deploy-prod
```

This enables:
- Resource limits (CPU/RAM)
- Persistent volumes
- Performance optimization
- Security settings
- Nginx reverse proxy

## 📚 Next Steps

1. **Read the full README.md** for detailed documentation
2. **Customize the pipeline** by editing source code in `src/`
3. **Add your API keys** in `.env` for external data enrichment
4. **Set up monitoring** with `make up-monitoring`
5. **Configure backups** with `make db-backup`

## 🤝 Need Help?

- **Check logs**: `make logs`
- **Verify health**: `make health`
- **View README**: Full documentation in `README.md`
- **Create issue**: Use GitHub issues for bugs/questions

---

**Total setup time: ~5 minutes** ⏱️