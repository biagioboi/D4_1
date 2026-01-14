from devp.crypt_util import *
import os
import subprocess

key = open("devp/filekey.key", "r").read()
def add_file(filename):
    if not os.path.isfile(filename):
        return 
    crypt(filename,key)
    output = subprocess.check_output("ipfs add -r "+ filename+"| awk '{print $2,$3}'| tee -a outputipfs.txt",shell=True)
    output = output.decode('utf-8').split(" ")
    cid_file = output[0]
    #nome_file = output[1].strip()
    subprocess.check_output("ipfs pin add -r "+cid_file,shell=True)
    return cid_file #,nome_file



def retrieve_file(cid,path_to_file):
    output = subprocess.check_output("ipfs get "+cid+" -o "+path_to_file,shell=True)
    decrypt(path_to_file,key) 



def add_files_from_folder(directory):
    cidList = []
    for filename in os.listdir(directory):
        f = os.path.join(directory,filename)
        #check if it is a file
        if os.path.isfile(f):
                res = add_file(f)
                if res != None:
                    cidList.append(res)
    return cidList


def delete_all_files_from_folder(directory):
    for filename in os.listdir(directory):
        f = os.path.join(directory,filename)
        #check if it is a file
        if os.path.isfile(f):
                os.remove(f)


def retrieve_files_from_cids_list(cids, destination_dir):
    count = 0
    for cid in cids:
        index = count + 1
        name_to_file = destination_dir + "/" + "model" + str(index)
        while os.path.exists(name_to_file):
            count += 1
            index = count + 1
            name_to_file = destination_dir + "/" + "model" + str(index)
        
        retrieve_file(cid, path_to_file=name_to_file)
        count += 1

def retriveModels(cids_list,dest_folder):
    os.makedirs(dest_folder,exist_ok=True)
    retrieve_files_from_cids_list(cids_list,dest_folder)


