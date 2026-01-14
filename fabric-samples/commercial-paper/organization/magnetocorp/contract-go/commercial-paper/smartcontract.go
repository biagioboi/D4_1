package chaincodefl

import (
	"encoding/json"
	"fmt"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// Contract provides functions for managing an Asset
type Contract struct {
	contractapi.Contract
}

// Asset describes basic details of what makes up a simple asset
type Asset struct {
	CID       string `json:"CID"`
	ModelType string `json:"ModelType"`
	Owner     string `json:"owner"`
	Accuracy  string `json:"accuracy"`
}

// InitLedger adds a base set of assets to the ledger
func (s *Contract) InitLedger(ctx contractapi.TransactionContextInterface) error {
	assets := []Asset{
		{CID: "00000", ModelType: "GLOBAL", Owner: "TEST", Accuracy: "0.0"},
	}

	for _, asset := range assets {
		assetJSON, err := json.Marshal(asset)
		if err != nil {
			return err
		}

		err = ctx.GetStub().PutState(asset.CID, assetJSON)
		if err != nil {
			return fmt.Errorf("failed to put to world state. %v", err)
		}
	}

	return nil
}

// CreateAsset issues a new asset to the world state with given details.
func (s *Contract) CreateAsset(ctx contractapi.TransactionContextInterface, cid string, ModelType string, owner string, Accuracy string) error {
	exists, err := s.AssetExists(ctx, cid)
	if err != nil {
		return err
	}
	if exists {
		return fmt.Errorf("the asset %s already exists", cid)
	}

	asset := Asset{
		CID:       cid,
		ModelType: ModelType,
		Owner:     owner,
		Accuracy:  Accuracy,
	}
	assetJSON, err := json.Marshal(asset)
	if err != nil {
		return err
	}

	return ctx.GetStub().PutState(cid, assetJSON)
}

// ReadAsset returns the asset stored in the world state with given id.
func (s *Contract) ReadAsset(ctx contractapi.TransactionContextInterface, cid string) (*Asset, error) {
	assetJSON, err := ctx.GetStub().GetState(cid)
	if err != nil {
		return nil, fmt.Errorf("failed to read from world state: %v", err)
	}
	if assetJSON == nil {
		return nil, fmt.Errorf("the asset %s does not exist", cid)
	}

	var asset Asset
	err = json.Unmarshal(assetJSON, &asset)
	if err != nil {
		return nil, err
	}

	return &asset, nil
}

// UpdateAsset updates an existing asset in the world state with provided parameters.
func (s *Contract) UpdateAsset(ctx contractapi.TransactionContextInterface, cid string, ModelType string, owner string, Accuracy string) error {
	exists, err := s.AssetExists(ctx, cid)
	if err != nil {
		return err
	}
	if !exists {
		return fmt.Errorf("the asset %s does not exist", cid)
	}

	// overwriting original asset with new asset
	asset := Asset{
		CID:       cid,
		ModelType: ModelType,
		Owner:     owner,
		Accuracy:  Accuracy,
	}
	assetJSON, err := json.Marshal(asset)
	if err != nil {
		return err
	}

	return ctx.GetStub().PutState(cid, assetJSON)
}

// DeleteAsset deletes an given asset from the world state.
func (s *Contract) DeleteAsset(ctx contractapi.TransactionContextInterface, id string) error {
	exists, err := s.AssetExists(ctx, id)
	if err != nil {
		return err
	}
	if !exists {
		return fmt.Errorf("the asset %s does not exist", id)
	}

	return ctx.GetStub().DelState(id)
}

// AssetExists returns true when asset with given ID exists in world state
func (s *Contract) AssetExists(ctx contractapi.TransactionContextInterface, id string) (bool, error) {
	assetJSON, err := ctx.GetStub().GetState(id)
	if err != nil {
		return false, fmt.Errorf("failed to read from world state: %v", err)
	}

	return assetJSON != nil, nil
}

// TransferAsset updates the owner field of asset with given id in world state.
/*
func (s *Contract) TransferAsset(ctx contractapi.TransactionContextInterface, id string, newOwner string) error {
	asset, err := s.ReadAsset(ctx, id)
	if err != nil {
		return err
	}

	asset.Owner = newOwner
	assetJSON, err := json.Marshal(asset)
	if err != nil {
		return err
	}

	return ctx.GetStub().PutState(id, assetJSON)
}
*/
// GetAllAssets returns all assets found in world state
func (s *Contract) GetAllAssets(ctx contractapi.TransactionContextInterface) ([]*Asset, error) {
	// range query with empty string for startKey and endKey does an
	// open-ended query of all assets in the chaincode namespace.
	resultsIterator, err := ctx.GetStub().GetStateByRange("", "")
	if err != nil {
		return nil, err
	}
	defer resultsIterator.Close()

	var assets []*Asset
	for resultsIterator.HasNext() {
		queryResponse, err := resultsIterator.Next()
		if err != nil {
			return nil, err
		}

		var asset Asset
		err = json.Unmarshal(queryResponse.Value, &asset)
		if err != nil {
			return nil, err
		}
		assets = append(assets, &asset)
	}

	return assets, nil
}

func (s *Contract) GetCids(ctx contractapi.TransactionContextInterface, owner string) ([]string, error) {
	queryString := fmt.Sprintf(`{"selector":{"owner":{"$regex":"(?i)^%s.*"}}}`, owner)
	// range query with empty string for startKey and endKey does an
	// open-ended query of all assets in the chaincode namespace.
	resultsIterator, err := ctx.GetStub().GetQueryResult(queryString)
	if err != nil {
		return nil, err
	}
	defer resultsIterator.Close()

	var res []string
	for resultsIterator.HasNext() {
		queryResponse, err := resultsIterator.Next()
		if err != nil {
			return nil, err
		}

		var asset Asset
		err = json.Unmarshal(queryResponse.Value, &asset)
		if err != nil {
			return nil, err
		}
		res = append(res, asset.CID)
	}
	print(res)

	return res, nil
}

func (s *Contract) GetModels(ctx contractapi.TransactionContextInterface, modelType string) ([]string, error) {
	queryString := fmt.Sprintf(`{"selector":{"ModelType":{"$regex":"(?i)^%s.*"}}}`, modelType)
	// range query with empty string for startKey and endKey does an
	// open-ended query of all assets in the chaincode namespace.
	resultsIterator, err := ctx.GetStub().GetQueryResult(queryString)
	if err != nil {
		return nil, err
	}
	defer resultsIterator.Close()

	var res []string
	for resultsIterator.HasNext() {
		queryResponse, err := resultsIterator.Next()
		if err != nil {
			return nil, err
		}

		var asset Asset
		err = json.Unmarshal(queryResponse.Value, &asset)
		if err != nil {
			return nil, err
		}
		res = append(res, asset.CID)
	}

	fmt.Println(res)
	return res, nil
}
