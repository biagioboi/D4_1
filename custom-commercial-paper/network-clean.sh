#!/bin/bash
#
# SPDX-License-Identifier: Apache-2.0

function _exit(){
    printf "Exiting:%s\n" "$1"
    exit -1
}

# Exit on first error, print all commands.
set -ev
set -o pipefail

# Where am I?
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

directory_path="${DIR}/organization/digibank/identity"

echo "Removing all subdirectories in $directory_path"
cd "$directory_path"
find . -maxdepth 1 -type d ! -name '.' -exec rm -rf {} +
echo "Done"

directory_path="${DIR}/organization/magnetocorp/identity"

echo "Removing all subdirectories in $directory_path"
cd "$directory_path"
find . -maxdepth 1 -type d ! -name '.' -exec rm -rf {} +
echo "Done"

cd "${DIR}"


export FABRIC_CFG_PATH="${DIR}/../config"

cd "${DIR}/../test-network/"

docker kill cliDigiBank cliMagnetoCorp logspout || true
./network.sh down

# remove any stopped containers
docker rm $(docker ps -aq)

