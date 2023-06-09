#!/bin/bash

cd "$(dirname "$0")"
source ./build.sh
docker push cr.yandex/${REGISTRY_ID}/manager:latest
