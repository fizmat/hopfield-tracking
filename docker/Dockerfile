FROM condaforge/mambaforge
ARG BUILDARCH

RUN --mount=type=cache,target=/var/lib/apt/lists \
    apt-get update && \
    apt-get install -y git
COPY environment.yml /environment.yml
RUN --mount=type=cache,target=/opt/conda/pkgs --mount=type=cache,target=/root/.cache/pip \
    mamba env create
RUN if [ $BUILDARCH = amd64 ]; then mamba install -n hopfield-tracking 'smac<2.0'; \
    else mamba run -n hopfield-tracking pip install 'smac<2.0'; fi;
COPY test_smac.py /test_smac.py
RUN mamba run -n hopfield-tracking python test_smac.py