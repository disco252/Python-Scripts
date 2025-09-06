HASHAUTO.py 

This is a Python script for Windows(I know, gross) designed to streamline the process of cracking multiple WiFi handshake hash files using Hashcat.

It optionally tranfers handshake files from a remote Raspberry Pi(pwnaogtchi) via SCP, then iterates through all .2200 and hc2200 hash files in the specified directory. 

The script tracks progress, prints real-time feedback from Hashcrack, and saves the cracked passwords to an output folder. 

Large-scale, repeated hash processing more practical and efficient. You are able to use all the basic hashcat commands with this, as well.

REQUIREMENTS:

Python 3

Hashcat

A valid hashcat wordlist AND rule file. 

Working SCP.

Set HASHCAT_PATH, HASHES_DIR, WORDLIST, OUTPUT_DIR, and (optionally) the SCP source in the script.

Launch HASHAUTO.py from command prompt, via python3 HASHAUTO.py after using "cd" to get to the directory. 
