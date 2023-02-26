import os, sys, shutil

source = "src/"


for root, dirs, files in os.walk(source): 
    
    for file in files:
        if file.startswith("prueba"): 
            try:
                os.remove(f"{source}/{file}") # remove files        
            except FileNotFoundError:
                print(f'not removed {source}/{file}')

print("Done")