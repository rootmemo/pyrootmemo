FROM quay.io/jupyter/base-notebook:latest

USER root

# Install system dependencies if needed
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

# Copy conda environment definition
COPY environment.yml /tmp/environment.yml

# Create/update the environment
RUN mamba env update -n base -f /tmp/environment.yml && \
    mamba clean --all -f -y

# Make sure notebooks use the environment
RUN python -m ipykernel install \
    --sys-prefix \
    --name rrmm \
    --display-name "Python (rrmm)"
