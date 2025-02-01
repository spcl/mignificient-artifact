#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BUILD_DIR=/work/serverless/2024/gpus/mignificient/build_116_release
APP=$1
DATA_DIR=../../../data/lukewarm-rtx-4070

mkdir -p ${DATA_DIR}/${APP}
cd ${DATA_DIR}/${APP}

for i in {0..9}; do

  mkdir -p sample_${i}
  cd sample_${i}

  ${BUILD_DIR}/orchestrator/orchestrator ${SCRIPT_DIR}/../config/orchestrator_seq.json ${BUILD_DIR}/../logs/devices.json > orchestrator_output.log 2>&1 &
  orchestrator_pid=$!

  sleep 1

  cp ${SCRIPT_DIR}/${APP}.json .

  ${BUILD_DIR}/invoker/bin/invoker ${APP}.json result.csv > invoker_output.log 2>&1


  kill -SIGINT ${orchestrator_pid}
  wait $orchestrator_pid

  cd ..
done
