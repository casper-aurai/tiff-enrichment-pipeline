# Container Publishing Guide

This guide explains how to publish the TIFF Enrichment Pipeline container images to both GitHub Container Registry (GHCR) and Docker Hub using GitHub Actions.

## 🚀 **Automated Publishing Setup**

### **What Gets Published**

The pipeline automatically publishes multi-architecture container images (`linux/amd64`, `linux/arm64`) to:

- **GitHub Container Registry**: `ghcr.io/casper-aurai/tiff-enrichment-pipeline`
- **Docker Hub**: `casper-aurai/tiff-enrichment-pipeline`

### **Publishing Triggers**

Images are automatically built and published when:

- ✅ **Push to main branch** → `latest` tag
- ✅ **Push to develop branch** → `develop` tag  
- ✅ **Create version tag** (e.g., `v1.2.3`) → version tags (`1.2.3`, `1.2`, `1`)
- ✅ **Pull requests** → test builds (not published)

## 🔧 **Setup Instructions**

### **1. Configure GitHub Secrets for Docker Hub**

To publish to Docker Hub, add these secrets to your GitHub repository:

1. Go to your repository → **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret** and add:

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `DOCKERHUB_USERNAME` | Your Docker Hub username | Used for Docker Hub login |
| `DOCKERHUB_TOKEN` | Your Docker Hub access token | Used for authentication |

#### **How to Get Docker Hub Access Token:**

1. Log in to [Docker Hub](https://hub.docker.com)
2. Go to **Account Settings** → **Security** → **Access Tokens**
3. Click **New Access Token**
4. Name: `GitHub Actions - TIFF Pipeline`
5. Permissions: **Read, Write, Delete**
6. Copy the generated token and add it as `DOCKERHUB_TOKEN` secret

### **2. GitHub Container Registry (GHCR) Setup**

GHCR publishing is automatically configured and uses the `GITHUB_TOKEN` that's provided by default. No additional setup required!

### **3. Repository Permissions**

Ensure your repository has the correct permissions:

1. Go to **Settings** → **Actions** → **General**
2. Under "Workflow permissions", select:
   - ✅ **Read and write permissions**
   - ✅ **Allow GitHub Actions to create and approve pull requests**

## 📦 **Using Published Images**

### **Quick Start with Published Images**

Instead of building locally, you can use the pre-built images:

```bash
# Using GitHub Container Registry (recommended)
docker pull ghcr.io/casper-aurai/tiff-enrichment-pipeline:latest

# Using Docker Hub
docker pull casper-aurai/tiff-enrichment-pipeline:latest

# Start pipeline with published images
make up-published

# Or with admin tools
make up-published-full
```

### **Available Tags**

| Tag | Description | Usage |
|-----|-------------|-------|
| `latest` | Latest stable release from main branch | Production |
| `develop` | Latest development build | Testing |
| `v1.2.3` | Specific version | Production (pinned) |
| `1.2` | Major.minor version | Production (auto-updates) |
| `1` | Major version | Production (major updates) |

### **Architecture Support**

All images support multiple architectures:
- ✅ **linux/amd64** (Intel/AMD 64-bit)
- ✅ **linux/arm64** (Apple Silicon, ARM servers)

Docker automatically pulls the correct architecture for your platform.

## 🛠️ **Advanced Usage**

### **Custom Registry Configuration**

Edit `docker-compose.published.yml` to use different registries:

```yaml
services:
  tiff-pipeline:
    # Use specific version
    image: ghcr.io/casper-aurai/tiff-enrichment-pipeline:v1.2.3
    
    # Or use Docker Hub
    # image: casper-aurai/tiff-enrichment-pipeline:latest
```

### **Private Registry**

To use a private registry, add authentication:

```bash
# Login to private registry
docker login your-registry.com

# Pull private image
docker pull your-registry.com/tiff-pipeline:latest
```

### **Verification with Cosign**

Images are signed with [Cosign](https://github.com/sigstore/cosign) for security:

```bash
# Install cosign
curl -O -L "https://github.com/sigstore/cosign/releases/latest/download/cosign-linux-amd64"
sudo mv cosign-linux-amd64 /usr/local/bin/cosign
sudo chmod +x /usr/local/bin/cosign

# Verify image signature
cosign verify --certificate-identity-regexp="https://github.com/casper-aurai/tiff-enrichment-pipeline" \
  --certificate-oidc-issuer="https://token.actions.githubusercontent.com" \
  ghcr.io/casper-aurai/tiff-enrichment-pipeline:latest
```

## 🏷️ **Release Process**

### **Creating a New Release**

1. **Update version number**:
   ```bash
   echo "1.2.3" > VERSION
   git add VERSION
   git commit -m "Bump version to 1.2.3"
   ```

2. **Create and push tag**:
   ```bash
   git tag v1.2.3
   git push origin v1.2.3
   ```

3. **GitHub Actions automatically**:
   - ✅ Builds multi-arch images
   - ✅ Publishes to GHCR and Docker Hub
   - ✅ Creates version tags (v1.2.3, 1.2.3, 1.2, 1)
   - ✅ Signs images with Cosign
   - ✅ Runs integration tests

### **Pre-release Testing**

Test development builds before releasing:

```bash
# Use develop branch builds
docker pull ghcr.io/casper-aurai/tiff-enrichment-pipeline:develop

# Or build from PR
docker pull ghcr.io/casper-aurai/tiff-enrichment-pipeline:pr-123
```

## 🔍 **Monitoring Build Status**

### **GitHub Actions**

Monitor builds at: `https://github.com/casper-aurai/tiff-enrichment-pipeline/actions`

### **Registry Status**

- **GitHub Container Registry**: `https://github.com/casper-aurai/tiff-enrichment-pipeline/pkgs/container/tiff-enrichment-pipeline`
- **Docker Hub**: `https://hub.docker.com/r/casper-aurai/tiff-enrichment-pipeline`

### **Image Metrics**

View download statistics:
- **GHCR**: Available in GitHub Insights
- **Docker Hub**: Available in Docker Hub repository page

## 🚨 **Troubleshooting**

### **Build Failures**

1. **Check GitHub Actions logs**:
   - Go to **Actions** tab in your repository
   - Click on the failed workflow
   - Review build logs for errors

2. **Common issues**:
   - ❌ **Docker Hub authentication failed** → Check `DOCKERHUB_TOKEN` secret
   - ❌ **Permission denied** → Check repository workflow permissions
   - ❌ **Disk space** → GitHub Actions has 14GB limit
   - ❌ **Rate limiting** → Docker Hub has pull rate limits

### **Image Pull Issues**

1. **Authentication required**:
   ```bash
   # Login to GitHub Container Registry
   echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
   
   # Login to Docker Hub
   docker login
   ```

2. **Image not found**:
   - Check tag exists in registry
   - Verify spelling of image name
   - Ensure build completed successfully

### **Local Development**

For local development, continue using the build approach:

```bash
# Local development (builds from source)
make build
make up

# Production deployment (uses published images)
make up-published
```

## 📋 **Best Practices**

1. **Use specific versions in production**:
   ```yaml
   image: ghcr.io/casper-aurai/tiff-enrichment-pipeline:v1.2.3
   ```

2. **Use `latest` for development**:
   ```yaml
   image: ghcr.io/casper-aurai/tiff-enrichment-pipeline:latest
   ```

3. **Pin major versions for stability**:
   ```yaml
   image: ghcr.io/casper-aurai/tiff-enrichment-pipeline:1
   ```

4. **Regular updates**:
   ```bash
   # Pull latest images
   make pull
   
   # Restart with new images
   make down && make up-published
   ```

---

**🎉 Your TIFF pipeline is now published and available worldwide!**

Anyone can now deploy your pipeline with a simple:
```bash
docker run -d ghcr.io/casper-aurai/tiff-enrichment-pipeline:latest
```