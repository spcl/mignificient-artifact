#!/bin/bash

if [ ! $# -eq 5 ]; then
    echo "Error: Please provide missing arguments"
    exit 1
fi

REPO_DIR=$1
BUILD_DIR=$2
DATA_DIR=$3
BENCHMARK=$4
GPU_JOB=$5

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
GPU_ID=0

device_configs=(1g 2g 3g 7g)
#device_configs=(1g)

#executor_configs=("baremetal" "sarus")
executor_configs=("baremetal")

#modes=("seq" "overlap" "overlap_memcpy" "full_overlap")
modes=("seq" "overlap" "overlap_memcpy")
#modes=("seq")

mkdir -p ${DATA_DIR}/function_switching/${BENCHMARK}
cd ${DATA_DIR}/function_switching/${BENCHMARK}

run_benchmark() {
    local executor_config=$1
    local mode=$2
    local job_number=$3

    echo "Begin ${device_config}, $1, $2"

    dir_name="${mode}_${executor_config}"
    mkdir -p "$dir_name"
    cd "$dir_name"

    srun --pty --oversubscribe -n 1 --jobid=${GPU_JOB} ${BUILD_DIR}/orchestrator/orchestrator ${SCRIPT_DIR}/config/orchestrator_${mode}.json ../devices.json > orchestrator_output.log 2>&1 &
    orchestrator_pid=$!

    cp ${SCRIPT_DIR}/${BENCHMARK}.json .

		for i in {0..1}; do

			new_path=$(jq --argjson idx $i --arg scriptdir "${SCRIPT_DIR}" '.inputs[$idx]."function-path" | gsub("SCRIPTDIR"; $scriptdir)' ${BENCHMARK}.json)

			new_value="$(pwd)/${BENCHMARK}-$i.txt"
			encoded_string=$(jq --argjson idx $i '.inputs[$idx]."input-payload" | gsub("[\\n\\t]"; "")' ${BENCHMARK}.json)

      val=$(echo "${encoded_string}" | jq --arg scriptdir "${SCRIPT_DIR}" --arg outfile "${new_value}" '. | fromjson | map_values(if type == "string" then gsub("SCRIPTDIR"; $scriptdir) | gsub("OUT"; $outfile) else . end)')
			#val=$(echo $val | jq --arg nv ${new_value} '."output-result" = $nv ')
			serialized_json=$(echo "$val" | jq 'tostring')

			jq --argjson idx $i --argjson fpath "${new_path}" --argjson payload "${serialized_json}" '.inputs[$idx]."input-payload" = $payload | .inputs[$idx]."function-path" = $fpath ' ${BENCHMARK}.json > temp.json && mv temp.json ${BENCHMARK}.json
		done

    ${BUILD_DIR}/invoker/bin/invoker ${BENCHMARK}.json result.csv > invoker_output.log 2>&1

    srun --pty --oversubscribe -n 1 --jobid=${GPU_JOB} killall -SIGINT orchestrator 
    wait $orchestrator_pid

    cd ..
}

for device_config in "${device_configs[@]}"; do

	mkdir -p ${device_config}
	cd ${device_config}

	# Configure MIG on the execution server
	srun --oversubscribe -n 1 --jobid=${GPU_JOB} ${REPO_DIR}/tools/configure-mig.sh ${GPU_ID} "$device_config"
	srun --oversubscribe -n 1 --jobid=${GPU_JOB} ${REPO_DIR}/tools/list-gpus.sh . ${GPU_ID}

	for executor_config in "${executor_configs[@]}"; do
			for mode in "${modes[@]}"; do
					run_benchmark "$executor_config" "$mode"
			done
	done

  cd ..
done
