# For finding latest versions of the base image see
# https://github.com/SwissDataScienceCenter/renkulab-docker
ARG RENKU_BASE_IMAGE=renku/renkulab-py:3.10-0.23.0
FROM ${RENKU_BASE_IMAGE} as builder

# RENKU_VERSION determines the version of the renku CLI
# that will be used in this image. To find the latest version,
# visit https://pypi.org/project/renku/#history.
ARG RENKU_VERSION=2.9.4

# Install renku from pypi or from github if a dev version
RUN if [ -n "$RENKU_VERSION" ] ; then \
    source .renku/venv/bin/activate ; \
    currentversion=$(renku --version) ; \
    if [ "$RENKU_VERSION" != "$currentversion" ] ; then \
        pip uninstall renku -y ; \
        gitversion=$(echo "$RENKU_VERSION" | sed -n "s/^[[:digit:]]\+\.[[:digit:]]\+\.[[:digit:]]\+\(rc[[:digit:]]\+\)*\(\.dev[[:digit:]]\+\)*\(+g\([a-f0-9]\+\)\)*\(+dirty\)*$/\4/p") ; \
        if [ -n "$gitversion" ] ; then \
            pip install --no-cache-dir --force "git+https://github.com/SwissDataScienceCenter/renku-python.git@$gitversion" ;\
        else \
            pip install --no-cache-dir --force renku==${RENKU_VERSION} ;\
        fi \
    fi \
fi

#             End Renku install section                #
########################################################
# Uncomment and adapt if code is to be included in the image
FROM ${RENKU_BASE_IMAGE}

# Uncomment and adapt if your R or python packages require extra linux (ubuntu) software
# e.g. the following installs apt-utils and vim; each pkg on its own line, all lines
# except for the last end with backslash '\' to continue the RUN line
#
# COPY src /code/src
# USER root
# RUN apt-get update && \
#    apt-get install -y --no-install-recommends \
#    apt-utils \
#    vim
# USER ${NB_USER}

### Install utilities
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    # libsm6 \
    # libxext6 \
    htop \ 
    git \ 
    wget
USER ${NB_USER}

# install the python dependencies
COPY env_humb.yml setup.py README.md /tmp/
ADD /src/ /tmp/src/
RUN mamba env update --name base --file /tmp/env_humb.yml && \
    mamba clean -y --all && \
    mamba env export -n "base" && \
    rm -rf ${HOME}/.renku/venv

COPY --from=builder ${HOME}/.renku/venv ${HOME}/.renku/venv