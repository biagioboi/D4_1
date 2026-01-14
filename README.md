# Demo – Federated Learning con supporto Blockchain per il salvataggio dei pesi dei modelli.

## Descrizione generale

Questo progetto rappresenta una demo sperimentale a supporto del Deliverable  
**D4.1 – "Analisi e progettazione dell’integrazione tra blockchain e IPFS per la memorizzazione in sicurezza dei parametri del modello di AI."** del progetto SEM.

La demo mostra:

- un processo di Federated Learning;
- il salvataggio e la tracciabilità delle versioni di un modello locale e globale su IPFS e su Hyperledger Fabric;
- l’uso di una blockchain per la registrazione di metadati associati alle versioni del modello.

## Federated Learning e attacchi di poisoning (`devp/`)

La cartella `devp/` contiene notebook Jupyter utilizzati per:

- simulare un processo di Federated Learning con più client;
- addestrare modelli di classificazione;
- introdurre attacchi di poisoning (label flipping, aggiornamenti malevoli);
- osservare l’andamento delle metriche del modello globale.

## Versioning del modello

Durante le simulazioni:

- i modelli vengono salvati a diversi round;
- le prestazioni del modello vengono monitorate nel tempo.

## Supporto Blockchain (`devp`)

La cartella `devp/` contiene il codice necessario a comunicare con la rete Fabric che verrà eseguita di seguito. Contiene le funzioni per:

- registrare asset su ledger;
- memorizzare hash, identificativi e owner delle versioni del modello;
- interrogare il ledger per finalità di tracciabilità.

La blockchain è utilizzata come registro di metadati.  
Il training del modello e i pesi sono gestiti off-chain.

## Requisiti

- `Python 3.11`
- `ipfs`
- `docker`
- `docker-compose`

## Setup ambiente Hyperledger Fabric
La copia del file createChannel è necessaria a causa di un bug presente nella versione di Fabric 2.3.3

```sh
curl -sSL https://bit.ly/2ysbOFE | bash -s -- 2.3.3 1.5.2
cp -r custom-commercial-paper fabric-samples
cp createChannel.sh fabric-samples/test-network/scripts
chmod +x fabric-samples/test-network/createChannel.sh
```

## Setup ambiente Python

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Avvio rete Fabric e deploy smart contract

```sh
cd fabric-samples/test-network
./network.sh down
./network.sh up createChannel -ca -c mychannel -s couchdb
./network.sh deployCC -c mychannel -ccn papercontract -ccp ../custom-commercial-paper/organization/digibank/contract-go/ -ccl go -ccep "OR('Org1MSP.peer','Org2MSP.peer')"
```

## Avvio demo applicativa

```sh
cd ../..
python main.py
```
