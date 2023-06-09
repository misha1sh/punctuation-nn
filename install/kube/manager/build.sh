#!/bin/bash
cd "$(dirname "$0")"

source ../env.sh
docker build . -f Dockerfile -t cr.yandex/$REGISTRY_ID/manager:latest
