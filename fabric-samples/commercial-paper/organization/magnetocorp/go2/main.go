package main

import (
	"flag"
	"fmt"
	f "go2/functions"
	"log"
	"os"
)

type UserPost struct {
	ID string `json:"id"`
}

type Asset struct {
	CID       string `json:"CID"`
	ModelType string `json:"ModelType"`
	Owner     string `json:"owner"`
}

type idAsset struct {
	CID string `json:"CID"`
}

var idUser, cid, ModelType, Accuracy string

func main() {
	flag.StringVar(&idUser, "id", "null", "User ID (required)")
	flag.StringVar(&cid, "cid", "null", "CID IPFS ")
	flag.StringVar(&ModelType, "modeltype", "null", "Model type (local, global) ")
	flag.StringVar(&Accuracy, "accuracy", "null", "Accuracy (0.00) ")
	createFlag := flag.Bool("create", false, "Invoke the \"Create\" function")
	getallFlag := flag.Bool("getall", false, "Invoke the \"Get all assets\" function")
	deleteFlag := flag.Bool("delete", false, "Invoke the \"Delete asset\" function")
	getModelsByTypeFlag := flag.Bool("getmodelsbytype", false, "Invoke the \"Get models by type\" function")
	flag.Parse()

	switch {
	case idUser == "null":
		fmt.Println("No id specified. Usage:")
		flag.PrintDefaults()
	case *createFlag:
		if cid == "null" || ModelType == "null" {
			fmt.Println("No cid or modeltype specified. Usage:")
			flag.PrintDefaults()
		} else {
			createAsset(idUser, cid, ModelType, Accuracy)
		}
	case *getallFlag:
		getAllItem()
	case *deleteFlag:
		if cid == "null" {
			fmt.Println("No cid specified. Usage:")
			flag.PrintDefaults()
		} else {
			deleteItem(cid)
		}
	case *getModelsByTypeFlag:
		if ModelType == "null" {
			fmt.Println("No modeltype specified. Usage:")
			flag.PrintDefaults()
		} else {
			getModelsByType()
		}
	default:
		fmt.Println("No function specified. Usage:")
		flag.PrintDefaults()
		os.Exit(1)
	}

}

func initLedger() {
	_, err := f.Inizialize(idUser)
	if err != nil {
		log.Println("errore:", err)
		os.Exit(1)
	}
}

func createAsset(idUser string, cid string, ModelType string, Accuracy string) {
	_, err := f.Create(idUser, cid, ModelType, idUser, Accuracy)
	if err != nil {
		log.Println("CREATE FAIL")
		log.Println("errore:", err)
		os.Exit(1)
	} else {
		log.Println("CREATE OK")
	}
}

func getAllItem() {
	log.Println("GET ALL START ")
	res, err := f.QueryAll(idUser)
	if err != nil {
		log.Println("errore:", err)
		os.Exit(1)
	}
	if res != nil {
		log.Println("RES:", string(res))
	} else {
		log.Println("null")
	}
}

func deleteItem(cid string) {
	log.Println("Delete item")
	res, err := f.DeleteAsset(idUser, cid)
	if err != nil {
		log.Println("errore:", err)
		os.Exit(1)
	}
	log.Println(string(res))

}

func getModelsByType() {
	log.Println("GET ALL START ")
	res, err := f.GetModelsByType(idUser, ModelType)
	if err != nil {
		log.Println("errore:", err)
		os.Exit(1)
	}
	if res != nil {
		log.Println("RES:", string(res))
	} else {
		log.Println("null")
	}
}

//--------------------------------------------------------------------------------------------
/*
func updateItem() {
	log.Println("UPDATE START")
	var appAsset Asset
	res, err := f.Update(idUser, appAsset.CID, appAsset.ModelType, appAsset.Owner)
	if err != nil {
		log.Println("errore:", err)
	}
	log.Println(string(res))

}
func getIDItem() {
	log.Println("GET START")
	var newIdAsset idAsset
	res, err := f.GetItemFromId(idUser, newIdAsset.CID)
	if err != nil {
		log.Println("errore:", err)
	}
	log.Println(string(res))

}



func createItem(w http.ResponseWriter, r *http.Request) {

	w.Header().Set("Content-Type", "application/json")

	log.Println("CREATE START")

	var newAsset Asset

	json.NewDecoder(r.Body).Decode(&newAsset)
	//prendi valori dal post e passali alla funzione
	assets = append(assets, newAsset)
	str, _ := json.MarshalIndent(newAsset, "", "")
	log.Println("ResultJSON: ", string(str))
	json.NewEncoder(w).Encode(&newAsset)

	_, err := f.Create(os.Args[1], newAsset.ID, newAsset.Temperature, newAsset.Humidity, newAsset.Owner)
	if err != nil {
		log.Println("CREATE FAIL")
	}
	log.Println("CREATE OK")

}
*/

/*
//POST LOGIC ok
func transferItem(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	log.Println("TRANSFER START")
	params := mux.Vars(r)
	for index, item := range assets {
		if item.ID == params["id"] {
			assets = append(assets[:index], assets[index+1:]...)

			var newAsset Asset
			_ = json.NewDecoder(r.Body).Decode(&newAsset)
			newAsset.ID = params["id"]

			_, err := f.Transfer(os.Args[1], newAsset.ID, newAsset.Owner)
			if err != nil {
				log.Println("TRANSFER FAIL")

			}
			str, _ := json.MarshalIndent(newAsset, "", "")
			log.Println("ResultJSON: ", string(str))

			assets = append(assets, newAsset)
			json.NewEncoder(w).Encode(&newAsset)
			log.Println("TRANSFER OK")
			return
		}
	}
	json.NewEncoder(w).Encode(assets)

}
*/

// GET LOGIC

/*
//POST LOGIC
func addUser(w http.ResponseWriter, r *http.Request) {

	log.Println("ADD USER START")
	w.Header().Set("Content-Type", "application/json")

	var userpost UserPost

	_ = json.NewDecoder(r.Body).Decode(&userpost)
	f.AddToWallet(userpost.ID)
	userposts = append(userposts, userpost)
	json.NewEncoder(w).Encode(&userposts)

	log.Printf("USER ID: %v ADDED", userpost.ID)

}

func removeUser(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	log.Println("REMOVE USER START")

	params := mux.Vars(r)
	for index, item := range userposts {
		if item.ID == params["id"] {
			id := params["id"]
			userposts = append(userposts[:index], userposts[index+1:]...)
			f.RemoveFromWallet(id)
			log.Printf("USER ID: %v REMOVED", id)
			break
		}
	}
	json.NewEncoder(w).Encode(userposts)
}

func getAllWallet(w http.ResponseWriter, r *http.Request) {

	w.Header().Set("Content-Type", "application/json")
	f.ListAll()
	json.NewEncoder(w).Encode(userposts)

	str, _ := json.MarshalIndent(userposts, "", "")
	log.Println("ResultJSON ID WALLET: ", string(str))
}
*/
