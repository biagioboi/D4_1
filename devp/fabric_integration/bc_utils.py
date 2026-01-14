import subprocess
import os


def network_start():
    pathnetwork = os.path.join(os.getcwd(), '../dev-fabric/commercial-paper')  
    cmd = "./contractdeploy.sh"
    result = subprocess.run(cmd, shell=True, cwd=pathnetwork, capture_output=True, text=True)


    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        #print(result.stdout)
        print(f"stderr: {result.stderr}")


def network_kill():
    pathnetwork = os.path.join(os.getcwd(), '../dev-fabric/commercial-paper')  
    cmd = "./network-clean.sh"
    result = subprocess.run(cmd, shell=True, cwd=pathnetwork, capture_output=True, text=True)
    
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        #print(result.stdout)
        print(f"stderr: {result.stderr}")