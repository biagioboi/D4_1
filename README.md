# Demo – Federated Learning con supporto Blockchain e rollback

## Descrizione generale

Questo progetto rappresenta una **demo sperimentale** a supporto del Deliverable **D4.2 – “Progettazione del meccanismo di rollback per tolleranza ad attacchi poisoning”** del progetto SEM.

L’obiettivo non è fornire una piattaforma completa o pronta per l’uso industriale, ma **mostrare in modo pratico e sperimentale** come:

- attacchi di poisoning possano degradare un processo di Federated Learning,
- la storia delle versioni del modello possa essere tracciata,
- e come un meccanismo di rollback concettuale possa essere supportato da tecnologie di tipo blockchain.

Il progetto va quindi inteso come **proof-of-concept / ambiente di test**, utile a validare idee architetturali e ipotesi di ricerca descritte nel documento D4.2.

## Contenuti del progetto

### 1. Federated Learning e attacchi di poisoning (cartella `devp/`)

La cartella `devp/` contiene diversi notebook Jupyter, utilizzati per:

- simulare un processo di **Federated Learning** con più client;
- addestrare modelli di classificazione (es. CIFAR-10);
- introdurre **attacchi di poisoning**, in particolare *label flipping* e aggiornamenti malevoli;
- osservare l’impatto degli attacchi sull’accuratezza e sulla loss del modello globale lungo più round.

Questa parte costituisce il **cuore sperimentale ML** del progetto e serve a mostrare quando e perché un rollback diventa necessario.

### 2. Concetto di versioning e rollback del modello

Durante le simulazioni:

- i modelli vengono salvati a diversi round;
- si osserva la degradazione delle metriche in presenza di attacchi;
- si individua manualmente o logicamente un punto “robusto” della storia del modello a cui tornare.

Il **rollback** non è completamente automatizzato ma è **sperimentato a livello concettuale**, coerentemente con quanto descritto nel D4.2, che si concentra sulla progettazione e non sull’implementazione finale.

### 3. Supporto Blockchain (cartella `dev-fabric/`)

La cartella `dev-fabric/` contiene una copia dell’ambiente **Hyperledger Fabric (dev mode)** utilizzata per sperimentare:

- la registrazione di asset su ledger;
- la memorizzazione di **hash, identificativi e owner** associati a versioni di modello;
- query di lettura dal ledger (tracciabilità, auditabilità).

La blockchain è usata **solo come registro di metadati**, non per il training né per la memorizzazione diretta dei pesi del modello, in linea con il D4.2.

### 4. Script di query e interrogazione

- `querybc`  
   Contiene comandi di esempio per interrogare il chaincode Fabric (init, query asset, query per owner, CID).
- `query couchdb`  
   Contiene una query Mango per interrogare CouchDB, usato come state database di Fabric.

Questi file servono come **supporto operativo** e dimostrativo.

## Cosa il progetto **è**

- Un **ambiente di test e sperimentazione**
- Un **supporto pratico** alle scelte progettuali del D4.2
- Un esempio di integrazione concettuale tra:
   - Federated Learning
   - attacchi di poisoning
   - versioning del modello
   - blockchain come registro immutabile

## Cosa il progetto **non è**

- Non è una piattaforma industriale completa
- Non è una implementazione finale del meccanismo di rollback
- Non include automazione completa, smart contract avanzati o IPFS pienamente integrato
- Non garantisce completezza, robustezza o conformità produttiva

## Obiettivo finale

Lo scopo del progetto è **rendere tangibile e sperimentabile** quanto descritto nel Deliverable D4.2, fornendo:

- esempi concreti di poisoning nel Federated Learning;
- una dimostrazione della necessità del rollback;
- un primo utilizzo di blockchain per tracciabilità e audit delle versioni del modello.

Il progetto può essere esteso in futuro con:

- decisioni automatiche di rollback,
- integrazione reale con IPFS,
- smart contract più avanzati,
- meccanismi di detection e game theory completamente integrati.

```sh
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```sh

```

