# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update
RUN apt-get install -y python3 python3-pip

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user that the app will run under.
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=wu_2025/,target=wu_2025/ \
    python3 -m pip install ./wu_2025

# Switch to the non-privileged user to run the application.
USER appuser

# Create a data volume
VOLUME ["/data"]
VOLUME ["/output"]

# Define environment variables
ENV INPUT=""
ENV OUTPUT=""

# Run the application
CMD python3 -m wu_2025 "/data/$INPUT" "/output/$OUTPUT"