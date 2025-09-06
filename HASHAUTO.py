import subprocess
import os
import re

# --- Configuration ---
HASHCAT_PATH = r"C:\Users\billo\OneDrive\Desktop\hashcat-6.2.6\hashcat.exe"
HASHES_DIR = r"C:\Users\billo\OneDrive\Desktop\hashcat-6.2.6"  # Also SCP output
WORDLIST = r"C:\Users\billo\OneDrive\Desktop\hashcat-6.2.6\test1.txt"
OUTPUT_DIR = r"C:\Users\billo\OneDrive\Desktop\hashcat-6.2.6\OUTPUT"
HASH_TYPE = "22000"
ATTACK_MODE = "0"

# --- Handshake Transfer Prompt and SCP Automation ---
response = input(
    "\nDo you want to transfer new handshakes from pi@10.0.0.2? (y/n): "
).strip().lower()
if response in ("y", "yes"):
    print("\nTransferring handshakes via SCP...")
    scp_cmd = [
        "scp",
        "-r",
        "pi@10.0.0.2:/pi/home/handshakes/*",
        HASHES_DIR
    ]
    subprocess.run(scp_cmd)
    print("Transfer complete.\n")

# --- Hashcat Batch Cracking Loop ---
skip_phrases = [
    "Dictionary cache building", "Guessing password", "Candidate.Engine.", "Candidates.#"
]
# Only process .22000 files to avoid parsing errors
hash_files = [
    f for f in os.listdir(HASHES_DIR)
    if f.endswith(".22000") or f.endswith(".hc22000")
]
for idx, hash_file in enumerate(hash_files, 1):
    print(f"\n--- Processing {hash_file} [{idx}/{len(hash_files)}] ---\n")
    print("[s]tatus [p]ause [b]ypass [c]heckpoint [f]inish [q]uit =>\n")
    full_hash_path = os.path.join(HASHES_DIR, hash_file)
    output_file = os.path.join(OUTPUT_DIR, f"cracked_{hash_file}")
    cmd = [
        HASHCAT_PATH,
        "-m", HASH_TYPE,
        "-a", ATTACK_MODE,
        full_hash_path,
        WORDLIST,
        "--outfile", output_file,
        "-D", "1,2",  # both gpu and cpu
        "-0",         # faster hardware optimized code paths on GPUs
        "-r", r"C:\Users\billo\OneDrive\Desktop\hashcat-6.2.6\rules\d3ad0ne.rule"
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace'
    )

    cracked_notice_printed = False
    last_recovered = "0"

    # Improved output formatting for readability
    for chunk in process.stdout:
        formatted = re.sub(
            r"(Counting lines in|Counted lines in|Parsed Hashes:|Sorting|Sorted|Removing|Removed|Comparing|Compared|Generating|Generated|Hashes:)",
            r"\n\1",
            chunk
        )
        for line in formatted.splitlines():
            if not line.strip():
                continue
            if any(phrase in line for phrase in skip_phrases):
                continue
            print(line.strip())

            # Check recovered status for dynamic notice
            m = re.search(r"Recovered\s*\.*:\s*(\d+)/", line)
            if m:
                last_recovered = m.group(1)
                if last_recovered == "0" and not cracked_notice_printed:
                    print("NOTICE: No hashes cracked yet. Check your attack mode, wordlist, or rules.")
                    cracked_notice_printed = True
                elif last_recovered != "0":
                    print(f"NOTICE: Success! {last_recovered} hashes cracked.")
                    cracked_notice_printed = True

    process.wait()
    if process.returncode == 0:
        print(f"\nNOTICE: Hashcat completed successfully for {hash_file}!\n")
    else:
        print(f"\nWARNING: Hashcat failed or exited with errors for {hash_file}.\n")
    print(f"\n=== Job complete for {hash_file} ===\n")

print("\nAll jobs completed!\n")
