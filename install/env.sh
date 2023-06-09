#!/bin/bash
REGISTRY_ID=$(yc container registry get --name reg --format json | jq .id -r)
