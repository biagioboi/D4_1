import os
from cryptography.fernet import Fernet
import base64

#need simmetric key

#chiave utilizzata
#key = open("filekey.key", "r").read()


def crypt(filepath,key):
    fernet = Fernet(key)
    with open(filepath, 'rb') as file:
        original = file.read()
    encrypted = fernet.encrypt(original)
    #print("io sono qui\n\n\n\n\n\n")
    #print(encrypted)
    with open(filepath,'wb') as encrypted_file:
        encrypted_file.write(encrypted)


def decrypt(filepath,key):
    fernet = Fernet(key)
    with open(filepath, 'rb') as file:
        encrypted = file.read().strip()
    #salvo temporaneamente il file
    decrypted = fernet.decrypt(encrypted)
    with open(filepath, 'wb') as decrypted_file:
        decrypted_file.write(decrypted)
     





