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

// Update can be used to update or prune the variable
func Issue(function, id string, issuer string, paperNumber string, issueDateTime string, maturityDateTime string, faceValue string) ([]byte, error) {

	log.Println("==============> SUBMIT ISSUE ")

	err := os.Setenv("DISCOVERY_AS_LOCALHOST", "true")
	if err != nil {
		return nil, fmt.Errorf("error setting DISCOVERY_AS_LOCALHOST environemnt variable: %v", err)
	}

	wallet, err := gateway.NewFileSystemWallet("../identity/user/operator/wallet")
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
	defer log.Println("==============> Disconnect from Fabric gateway ")
	defer log.Println("==============> Issue program complete ")
	defer gw.Close()

	log.Println("==============> Use mychannel ")

	network, err := gw.GetNetwork("mychannel")
	if err != nil {
		return nil, fmt.Errorf("failed to get network: %v", err)
	}

	contract := network.GetContract("papercontract")

	// const (
	// 	issuer           string = "MagnetoCorp"
	// 	paperNumber      string = "00001"
	// 	issueDateTime    string = "2020-05-31"
	// 	maturityDateTime string = "2020-11-30"
	// 	faceValue        string = "5000000"
	// )

	result, err := contract.SubmitTransaction(function, issuer, paperNumber, issueDateTime, maturityDateTime, faceValue)
	if err != nil {
		return result, fmt.Errorf("failed to Submit transaction: %v", err)
	}
	fmt.Println(string(result))

	log.Printf("==============> Process issue transaction response. %v", string(result))

	result, err = contract.SubmitTransaction("query")
	if err != nil {
		return nil, fmt.Errorf("failed to evaluate transaction: %v", err)
	}
	log.Printf("==============> Process query transaction response: %v", result)

	return result, err
}
