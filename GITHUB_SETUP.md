# GitHub Repository Setup Instructions

## 1. Create Private Repository on GitHub

Go to https://github.com/new and create a new repository:

- **Repository name**: `latence-python` (or `python-sdk`)
- **Description**: Official Python SDK for Latence AI API Gateway
- **Visibility**: ✅ **Private** (for now)
- **Initialize**: ❌ Do NOT initialize with README, .gitignore, or license (we already have these)

## 2. Push Local Repository

After creating the repository, run these commands:

```bash
cd /workspace/latence-python

# Add remote (replace with your actual GitHub URL)
git remote add origin https://github.com/latenceai/latence-python.git

# Push to GitHub
git push -u origin main
```

## 3. Configure Repository Settings

### Branch Protection
Go to: **Settings → Branches → Add rule**

- Branch name pattern: `main`
- ✅ Require pull request reviews before merging
- ✅ Require status checks to pass before merging
  - Select: `test`

### PyPI Trusted Publishing (OIDC)

The publish workflow uses **Trusted Publishing** (OIDC), so no API token secret is needed.
Before your first release, configure a **pending trusted publisher** on PyPI:

1. Go to https://pypi.org/manage/account/publishing/
2. Fill in:
   - **PyPI project name**: `latence`
   - **Owner**: `latenceai`
   - **Repository**: `latence-python`
   - **Workflow name**: `publish.yml`
   - **Environment name**: `pypi`
3. Click **Add**

Then create a GitHub Environment named `pypi` in the repository:

1. Go to: **Settings → Environments → New environment**
2. Name it `pypi`
3. Optionally enable **Required reviewers** for manual approval on each release

### Secrets (for integration tests)
Go to: **Settings → Secrets and variables → Actions**

- `STAGING_API_URL` - Base URL for the staging API gateway
- `STAGING_API_KEY` - API key for staging integration tests

### Topics
Go to: **Settings → General → Topics**

Add topics for discoverability:
- `python`
- `sdk`
- `api-client`
- `latence`
- `embeddings`
- `nlp`
- `document-processing`

## 4. Make Repository Public (When Ready)

Once everything is perfect and you're ready to release:

1. Go to: **Settings → General → Danger Zone**
2. Click "Change repository visibility"
3. Select "Make public"
4. Confirm

## 5. Create First Release

Go to: **Releases → Create a new release**

- **Tag**: `v0.1.0`
- **Title**: `v0.1.0 - Initial Release`
- **Description**: Copy from README highlights
- Publish release → This will automatically trigger PyPI publishing workflow

## Repository Structure

```
latenceai/latence-python/
├── .github/
│   └── workflows/
│       ├── test.yml          # CI tests
│       └── publish.yml       # PyPI publishing
├── src/latence/              # Package source
├── tests/                    # Test suite
├── notebooks/                # Tutorial notebooks
├── README.md                 # Main documentation
├── pyproject.toml            # Package configuration
├── LICENSE                   # MIT License
├── CONTRIBUTING.md           # Contribution guide
└── .gitignore               # Git ignore rules
```

## Quick Commands Reference

```bash
# Clone repository (after creating on GitHub)
git clone https://github.com/latenceai/latence-python.git

# Install in development mode
cd latence-python
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Check types
mypy src/latence

# Build package
python -m build

# Install from local build
pip install dist/latence-0.1.0-py3-none-any.whl
```

## PyPI Package Name

The package will be published as **`latence`** on PyPI:

```bash
pip install latence
```

```python
from latence import Latence

client = Latence(api_key="your_api_key")
```

## Note

Keep the repository **private** until:
- ✅ All features tested and verified
- ✅ Documentation complete
- ✅ README polished
- ✅ Example notebooks working
- ✅ First version ready for public release
