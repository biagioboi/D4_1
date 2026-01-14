#!/bin/bash


./network-starter.sh

# Cartella contenente lo script principale


gnome-terminal -- bash -c "cd organization/digibank ; source digibank.sh ; ./digibankStart.sh; gnome-terminal -- bash -c 'cd ../magnetocorp ; source magnetocorp.sh ; ./magnetocorpStart.sh ; ./commit.sh'"
