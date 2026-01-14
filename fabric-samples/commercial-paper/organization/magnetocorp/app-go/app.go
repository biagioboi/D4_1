/*
Copyright 2020 IBM All Rights Reserved.

SPDX-License-Identifier: Apache-2.0
*/

package main

import (
	f "app-go/functions"
	"app-go/user"
	"log"
	"os"
)

func main() {

	id := os.Args[1]
	function := os.Args[2]

	log.Println("Usage: app.go id function ")
	switch function {
	case "add":

		err := user.AddToWallet(id)

		if err != nil {
			log.Printf("Error: unable to put identity %v in wallet", id)
		} else {
			log.Printf("Succes: put identity %v in wallet", id)
		}

	case "remove":

		err := user.RemoveToWallet(id)

		if err != nil {
			log.Printf("Error: unable to delete identity %v in wallet", id)
		} else {
			log.Printf("Succes: delete identity %v in wallet", id)
		}
	case "list":
		//...
		err := user.ListAll()
		if err != nil {
			log.Printf("Error: unable to list identity %v ", err)
		}

	case "issue":

		log.Println("==============>Start ISSUE ")
		var (
			issuer           string = "MagnetoCorp"
			paperNumber      string = "00001"
			issueDateTime    string = "2020-05-31"
			maturityDateTime string = "2020-11-30"
			faceValue        string = "5000000"
		)
		result, err := f.Issue(function, id, issuer, paperNumber, issueDateTime, maturityDateTime, faceValue)
		if err != nil {
			log.Fatalf("error: %v", err)
		}
		log.Println(string(result))

	case "buy":
		log.Println("==============>Start BUY")

		var (
			issuer           string = "MagnetoCorp"
			paperNumber      string = "00001"
			currentOwner     string = "MagnetoCorp"
			newOwner         string = "DigiBank"
			price            string = "4900000"
			purchaseDateTime string = "2020-05-31"
		)

		result, err := f.Buy(function, id, issuer, paperNumber, currentOwner, newOwner, price, purchaseDateTime)
		if err != nil {
			log.Fatalf("error: %v", err)
		}
		log.Println(string(result))

	case "redeem":

		log.Println("==============>Start REDEEM")

		var (
			issuer         string = "MagnetoCorp"
			paperNumber    string = "00001"
			redeemingOwner string = "DigiBank"
			redeenDateTime string = "2020-11-30"
		)

		result, err := f.Redeem(function, id, issuer, paperNumber, redeemingOwner, redeenDateTime)
		if err != nil {
			log.Fatalf("error: %v", err)
		}
		log.Println(string(result))
	case "query":
		log.Println("==============Start Query ==============")

		result, err := f.Query("Query", id)
		if err != nil {
			log.Fatalf("error: %v", err)
		}
		log.Println(string(result))

	default:
		log.Println("Usage: app.go id function ")
		log.Println("you have to type like this: app.go user1 addwallet")

		log.Println("function : issue , buy , redeem, add-list-remove  ")

	}
	// if function != "issue" || function != "buy" || function != "redeem" {
	// 	log.Println("function : issue , buy , redeem")
	// 	os.Exit(1)
	// }

}
