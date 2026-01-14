import subprocess
import json
import os
from .. import ipfs_files_util as ipfs




def getallAssets(id):
    path = os.path.join(os.getcwd(),'fabric-samples/commercial-paper/digibank/magnetocorp/go2')
    cmd = 'go run main.go'+" -id "+id+" -getall"
    result = subprocess.run(cmd, shell=True, cwd=path, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print("Success")

    output_lines = result.stderr.splitlines()
    res_line = None
    for line in output_lines:
        if "RES:" in line:
            res_line = line.strip()
            break
    if res_line is None:
            print("No output found")
            return None
    else:
        json_data = res_line.split('RES: ')[-1]

    return json.loads(json_data)

def createAsset(id,cid,modeltype,accuracy = 0):
    path = os.path.join(os.getcwd(),'fabric-samples/commercial-paper/organization/digibank/go2')
    cmd = 'go run main.go'+" -id "+id+" -create -cid "+cid+" -modeltype "+modeltype +" -accuracy "+str(accuracy)
    result = subprocess.run(cmd, shell=True, cwd=path, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    else:
        return True

def deleteAsset(id,cid):
    path = os.path.join(os.getcwd(),'fabric-samples/commercial-paper/organization/digibank/go2')
    cmd = 'go run main.go'+" -id "+id+" -delete -cid "+cid
    result = subprocess.run(cmd, shell=True, cwd=path, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print("Success")

def deleteAssetsFromCidsList(id, cids):
        for cid in cids:
            deleteAsset(id, cid)


def getModelsByType(id,modeltype):
    path = os.path.join(os.getcwd(),'fabric-samples/commercial-paper/organization/digibank/go2')
    cmd = 'go run main.go'+" -id "+id+" -getmodelsbytype -modeltype "+modeltype
    result = subprocess.run(cmd, shell=True, cwd=path, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print("Success")

    output_lines = result.stderr.splitlines()
    res_line = None
    for line in output_lines:
        if "RES:" in line:
            res_line = line.strip()
            break
    if res_line is None:
            print("No output found")
            return None
    else:
        json_data = res_line.split('RES: ')[-1]
        return json.loads(json_data)


def addModels(id,models_dir,accuracy=0):
    cids_list=[]
    cid = None
    for file in os.listdir(models_dir):
        if file != ".ipynb_checkpoints":
            if file.__contains__("global"):
                cid = ipfs.add_file(os.path.join(models_dir,file))
                res = createAsset(id,cid,"global",accuracy)
                if res == False:
                    return None
                os.remove(os.path.join(models_dir,file))
            else:
                cid = ipfs.add_file(os.path.join(models_dir,file))
                createAsset(id,cid,"client")
                os.remove(os.path.join(models_dir,file))
            cids_list.append(cid)
    return cids_list

# da testare
def addModel(id,file,accuracy=0):
    cid = None
    if file != ".ipynb_checkpoints":
            if file.__contains__("global"):
                cid = ipfs.add_file(file)
                res = createAsset(id,cid,"global",accuracy)
                if res == False:
                    return None
                os.remove(file)
            else:
                cid = ipfs.add_file(file)
                createAsset(id,cid,"client")
    return cid



 
 
     
