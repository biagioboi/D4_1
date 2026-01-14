/*
Copyright 2020 IBM All Rights Reserved.

SPDX-License-Identifier: Apache-2.0
*/

package functions

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/hyperledger/fabric-sdk-go/pkg/core/config"
	"github.com/hyperledger/fabric-sdk-go/pkg/gateway"
)

// Query can be used to read the latest value of a variable
func Query(function, id string) ([]byte, error) {

	log.Println("==============> SUBMIT QUERY ")

	err := os.Setenv("DISCOVERY_AS_LOCALHOST", "true")
	if err != nil {
		return nil, fmt.Errorf("error setting DISCOVERY_AS_LOCALHOST environemnt variable: %v", err)
	}

	log.Println("==============> I'm building the wallet...")

	path := "../identity/user/" + id
	path2 := path + "/wallet"
	log.Println("Wallet path: ", path2)

	wallet, err := gateway.NewFileSystemWallet(path2)
	if err != nil {
		return nil, fmt.Errorf("failed to create wallet: %v", err)
	}

	if !wallet.Exists(id) {
		err := populateWallet(wallet, id)
		if err != nil {
			return nil, fmt.Errorf("failed to populate wallet contents: %v", err)
		}
	}

	ccpPath := filepath.Join(
		"..",
		"..",
		"..",
		"..",
		"test-network",
		"organizations",
		"peerOrganizations",
		"org2.example.com",
		"connection-org2.yaml",
	)

	gw, err := gateway.Connect(
		gateway.WithConfig(config.FromFile(filepath.Clean(ccpPath))),
		gateway.WithIdentity(wallet, id),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to gateway: %v", err)
	}
	defer log.Println("==============> Disconnect from Fabric gateway")
	defer log.Println("==============> Query program complete")
	defer gw.Close()
	log.Println("==============> Use mychannel")

	network, err := gw.GetNetwork("mychannel")
	if err != nil {
		return nil, fmt.Errorf("failed to get network: %v", err)
	}

	log.Println("==============> Get papercontract smart contract")

	contract := network.GetContract("papercontract")

	result, err := contract.EvaluateTransaction(function)
	if err != nil {
		return nil, fmt.Errorf("failed to evaluate transaction: %v", err)
	}
	fmt.Println(string(result))
	log.Printf("==============> Process query response: %v", string(result))
	return result, err
}
