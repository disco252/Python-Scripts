"""
Unified OSINT Search Tool v3
Integrates: Maigret, Sherlock, SpiderFoot, theHarvester, WhatsMyName, PhoneInfoga
"""

import os
import sys
import subprocess
import json
import time
import re
from datetime import datetime
from pathlib import Path
import argparse

class OsintMaster:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.results_dir = self.base_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Tool paths (local directories for tools that aren't installed via pip)
        self.tools = {
            'maigret': self.base_dir / "maigret",
            'sherlock': self.base_dir / "sherlock",
            'spiderfoot': self.base_dir / "spiderfoot" / "sf.py",
            'whatsmyname': self.base_dir / "WhatsMyName-Python" / "whatsmyname.py"
        }
        
    def print_banner(self):
        banner = r"""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   ██████╗ ███████╗██╗███╗   ██╗████████╗    ███╗   ███╗ █████╗        ║
║  ██╔═══██╗██╔════╝██║████╗  ██║╚══██╔══╝    ████╗ ████║██╔══██╗       ║
║  ██║   ██║███████╗██║██╔██╗ ██║   ██║       ██╔████╔██║███████║       ║
║  ██║   ██║╚════██║██║██║╚██╗██║   ██║       ██║╚██╔╝██║██╔══██║       ║
║  ╚██████╔╝███████║██║██║ ╚████║   ██║       ██║ ╚═╝ ██║██║  ██║       ║
║   ╚═════╝ ╚══════╝╚═╝╚═╝  ╚═══╝   ╚═╝       ╚═╝     ╚═╝╚═╝  ╚═╝       ║
║                                                                       ║
║              Unified OSINT Automation Suite v3                        ║
║        Maigret • Sherlock • SpiderFoot • theHarvester                 ║
║           WhatsMyName • PhoneInfoga                                   ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""
        print(banner)
    
    def extract_urls_from_text(self, text):
        """Extract all URLs from text"""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        return re.findall(url_pattern, text)
    
    def extract_emails_from_text(self, text):
        """Extract email addresses from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(email_pattern, text)
    
    def sanitize_phone_number(self, phone):
        """Sanitize phone number for filename"""
        return phone.replace('+', '').replace(' ', '').replace('-', '').replace('(', '').replace(')', '')
    
    def run_maigret(self, username, output_dir):
        """Run Maigret username search"""
        print(f"\n[*] Running Maigret on: {username}")
        
        output_txt = output_dir / f"{username}_maigret_output.txt"
        output_results = output_dir / f"{username}_maigret_results.txt"
        
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        cmd = [
            sys.executable, "-m", "maigret",
            username,
            "--folderoutput", str(output_dir),
            "--timeout", "15",
            "--no-color",
            "--no-progressbar"
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300,
                env=env,
                encoding='utf-8',
                errors='replace'
            )
            
            with open(output_txt, 'w', encoding='utf-8', errors='replace') as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n\n=== STDERR ===\n")
                    f.write(result.stderr)
            
            # Extract found accounts
            found_accounts = []
            for line in result.stdout.split('\n'):
                if '[+]' in line and 'http' in line:
                    parts = line.split('[+]')
                    if len(parts) > 1:
                        found_accounts.append(parts[1].strip())
            
            if found_accounts:
                with open(output_results, 'w', encoding='utf-8') as f:
                    f.write(f"Found {len(found_accounts)} accounts for username: {username}\n")
                    f.write("="*70 + "\n\n")
                    for account in found_accounts:
                        f.write(f"{account}\n")
                
                print(f"[+] Maigret completed - Found {len(found_accounts)} accounts!")
            else:
                print(f"[+] Maigret completed")
            
            html_files = list(output_dir.glob("*.html"))
            if html_files:
                print(f"    HTML report: {html_files[0].name}")
            
            return True
                    
        except subprocess.TimeoutExpired:
            print(f"[-] Maigret timed out")
            return False
        except Exception as e:
            print(f"[-] Maigret error: {e}")
            return False
    
    def run_sherlock(self, username, output_dir):
        """Run Sherlock username search"""
        print(f"\n[*] Running Sherlock on: {username}")
        
        output_log = output_dir / f"{username}_sherlock_output.txt"
        output_results = output_dir / f"{username}_sherlock_results.txt"
        
        # Try multiple methods
        sherlock_commands = [
            ["sherlock", username, "--timeout", "10", "--print-found"],
            [sys.executable, "-m", "sherlock_project", username, "--timeout", "10"],
            [sys.executable, str(self.tools['sherlock'] / "sherlock_project" / "sherlock.py"), username, "--timeout", "10"],
        ]
        
        for i, cmd in enumerate(sherlock_commands):
            try:
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=180,
                    encoding='utf-8',
                    errors='replace'
                )
                
                with open(output_log, 'w', encoding='utf-8') as f:
                    f.write(f"Attempt {i+1} - Command: {' '.join(str(c) for c in cmd)}\n")
                    f.write("="*70 + "\n\n")
                    f.write(result.stdout)
                    if result.stderr:
                        f.write("\n\n=== STDERR ===\n")
                        f.write(result.stderr)
                
                if result.returncode == 0 or "http" in result.stdout.lower():
                    urls = self.extract_urls_from_text(result.stdout)
                    
                    if urls:
                        with open(output_results, 'w', encoding='utf-8') as f:
                            f.write(f"Found {len(urls)} accounts for username: {username}\n")
                            f.write("="*70 + "\n\n")
                            for url in urls:
                                f.write(f"{url}\n")
                        
                        print(f"[+] Sherlock completed - Found {len(urls)} accounts!")
                    else:
                        print(f"[+] Sherlock completed - check {output_log.name}")
                    
                    return True
                
            except FileNotFoundError:
                continue
            except Exception as e:
                continue
        
        print(f"[!] Sherlock not available")
        print(f"    Install with: pip install sherlock-project")
        with open(output_log, 'w', encoding='utf-8') as f:
            f.write("Sherlock not installed or not accessible.\n")
            f.write("Install with: pip install sherlock-project\n")
        return False
    
    def run_whatsmyname(self, username, output_dir):
        """Run WhatsMyName username search"""
        print(f"\n[*] Running WhatsMyName on: {username}")
        
        wmn_path = self.tools['whatsmyname']
        
        if not wmn_path.exists():
            print(f"[-] WhatsMyName not found")
            return False
        
        output_txt = output_dir / f"{username}_whatsmyname_output.txt"
        output_results = output_dir / f"{username}_whatsmyname_results.txt"
        
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        wmn_dir = wmn_path.parent
        cmd = [sys.executable, str(wmn_path), "-u", username]
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=180, 
                cwd=str(wmn_dir),
                env=env,
                encoding='utf-8',
                errors='replace'
            )
            
            with open(output_txt, 'w', encoding='utf-8', errors='replace') as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n\n=== STDERR ===\n")
                    f.write(result.stderr)
            
            urls = self.extract_urls_from_text(result.stdout)
            
            if urls:
                with open(output_results, 'w', encoding='utf-8') as f:
                    f.write(f"Found {len(urls)} accounts for username: {username}\n")
                    f.write("="*70 + "\n\n")
                    for url in urls:
                        f.write(f"{url}\n")
                print(f"[+] WhatsMyName completed - Found {len(urls)} accounts!")
            
            csv_files = list(wmn_dir.glob(f"*{username}*.csv"))
            if csv_files:
                import shutil
                for csv_file in csv_files:
                    dest = output_dir / csv_file.name
                    shutil.copy(str(csv_file), str(dest))
                    print(f"    CSV: {csv_file.name}")
            
            if "getaddrinfo failed" in result.stderr or "ConnectionError" in result.stderr:
                print(f"[!] WhatsMyName network error - Check DNS/internet connection")
                return False
            
            if not urls and not csv_files:
                print(f"[+] WhatsMyName completed - check {output_txt.name}")
            
            return True
            
        except Exception as e:
            print(f"[-] WhatsMyName error: {e}")
            return False
    
    def check_theharvester_setup(self):
        """Check theHarvester installation - Uses installed package"""
        try:
            # Test if theHarvester is installed and working
            result = subprocess.run(
                [sys.executable, "-m", "theHarvester", "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 or "theHarvester" in result.stdout.lower():
                return {'status': 'ok'}
            else:
                return {
                    'status': 'error',
                    'message': 'theHarvester installed but not responding correctly'
                }
                
        except subprocess.TimeoutExpired:
            return {
                'status': 'error',
                'message': 'theHarvester check timed out'
            }
        except FileNotFoundError:
            return {
                'status': 'missing',
                'message': 'theHarvester not installed',
                'install_cmd': 'pip install theHarvester'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'theHarvester check failed: {str(e)}',
                'install_cmd': 'pip install theHarvester'
            }
    
    def run_theharvester(self, domain, output_dir):
        """Run theHarvester for domain/email search - Uses installed package"""
        print(f"\n[*] Running theHarvester on: {domain}")
        
        # Check if theHarvester is installed
        setup = self.check_theharvester_setup()
        
        if setup['status'] != 'ok':
            print(f"[-] {setup['message']}")
            if 'install_cmd' in setup:
                print(f"    Install with: {setup['install_cmd']}")
                print(f"    Or run: cd theHarvester && pip install .")
            return False
        
        # Prepare output files
        output_txt = output_dir / f"{domain.replace('.', '_')}_harvester_output.txt"
        output_results = output_dir / f"{domain.replace('.', '_')}_harvester_results.txt"
        
        # Use installed package via python -m
        cmd = [
            sys.executable, "-m", "theHarvester",
            "-d", domain,
            "-b", "bing,google",
            "-l", "100"
        ]
        
        try:
            print(f"    Searching Bing and Google...")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300,
                encoding='utf-8',
                errors='replace'
            )
            
            # Save full output
            with open(output_txt, 'w', encoding='utf-8', errors='replace') as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n\n=== STDERR ===\n")
                    f.write(result.stderr)
            
            # Check for critical errors
            if "ModuleNotFoundError" in result.stderr or "ImportError" in result.stderr:
                print(f"[-] theHarvester has dependency issues")
                print(f"    Check: {output_txt.name}")
                print(f"    Try: pip install --upgrade theHarvester")
                return False
            
            # Extract results from output
            emails = self.extract_emails_from_text(result.stdout)
            
            # Extract hosts (subdomains/IPs)
            hosts = []
            for line in result.stdout.split('\n'):
                # Look for lines with hosts/IPs
                if re.search(r'\d+\.\d+\.\d+\.\d+', line):  # IP addresses
                    ips = re.findall(r'\d+\.\d+\.\d+\.\d+', line)
                    hosts.extend(ips)
                # Look for subdomain patterns
                match = re.search(rf'([a-zA-Z0-9][-a-zA-Z0-9]*\.)*{re.escape(domain)}', line)
                if match:
                    hosts.append(match.group(0))
            
            # Save parsed results
            if emails or hosts:
                with open(output_results, 'w', encoding='utf-8') as f:
                    f.write(f"theHarvester Results for: {domain}\n")
                    f.write("="*70 + "\n\n")
                    
                    if emails:
                        unique_emails = sorted(set(emails))
                        f.write(f"EMAILS FOUND ({len(unique_emails)}):\n")
                        f.write("-"*70 + "\n")
                        for email in unique_emails:
                            f.write(f"{email}\n")
                        f.write("\n")
                    
                    if hosts:
                        unique_hosts = sorted(set(hosts))
                        f.write(f"HOSTS/IPs FOUND ({len(unique_hosts)}):\n")
                        f.write("-"*70 + "\n")
                        for host in unique_hosts:
                            f.write(f"{host}\n")
                
                print(f"[+] theHarvester completed!")
                print(f"    Emails: {len(set(emails))}, Hosts/IPs: {len(set(hosts))}")
            else:
                print(f"[+] theHarvester completed - No results found")
                print(f"    Full output: {output_txt.name}")
            
            return True
            
        except subprocess.TimeoutExpired:
            print(f"[-] theHarvester timed out after 5 minutes")
            return False
        except Exception as e:
            print(f"[-] theHarvester error: {e}")
            return False
    
    def check_phoneinfoga_setup(self):
        """Check PhoneInfoga installation"""
        # Check for local phoneinfoga.exe in subfolder first
        local_paths = [
            self.base_dir / "phoneinfoga" / "phoneinfoga.exe",  # In subfolder
            self.base_dir / "phoneinfoga.exe",  # In main folder
        ]
        
        for local_exe in local_paths:
            if local_exe.exists():
                try:
                    result = subprocess.run(
                        [str(local_exe), "version"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0 or "phoneinfoga" in result.stdout.lower():
                        return {'status': 'ok', 'type': 'local binary', 'path': str(local_exe)}
                except Exception as e:
                    continue
        
        # Try system PATH
        try:
            result = subprocess.run(
                ["phoneinfoga", "version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 or "phoneinfoga" in result.stdout.lower():
                return {'status': 'ok', 'type': 'binary', 'path': 'phoneinfoga'}
                
        except FileNotFoundError:
            pass
        except Exception as e:
            pass
        
        return {
            'status': 'missing',
            'message': 'PhoneInfoga not installed',
            'install_cmd': 'Download from: https://github.com/sundowndev/phoneinfoga/releases'
        }
    
    def validate_phone_search_urls(self, urls, phone_number, max_check=45):
        """
        Validate Google search URLs to find which have actual results
        Returns only URLs that contain search results
        """
        validated = []
        
        try:
            import requests
        except ImportError:
            print(f"    [!] requests library not found - skipping validation")
            print(f"        Install with: pip install requests")
            return []
        
        # Categorize URLs
        categories = {
            'facebook.com': 'Facebook',
            'twitter.com': 'Twitter',
            'linkedin.com': 'LinkedIn',
            'instagram.com': 'Instagram',
            'pastebin.com': 'Pastebin',
            'whosenumber': 'Caller ID',
            'whocalled': 'Caller ID',
            'Phone+Fraud': 'Fraud Reports',
            'yellowpages': 'Directory',
            'receive-sms': 'Temp Number',
            'disposable': 'Temp Number',
            'numinfo': 'Phone Lookup',
            'sync.me': 'Phone Lookup'
        }
        
        # Headers to mimic browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        print(f"    Validating URLs (checking {min(len(urls), max_check)} of {len(urls)} URLs)...")
        print(f"    This may take 30-60 seconds...")
        
        checked = 0
        for url in urls[:max_check]:  # Limit to avoid rate limiting
            checked += 1
            if checked % 5 == 0:
                print(f"       • Progress: {checked}/{min(len(urls), max_check)}...")
            
            try:
                # Determine category
                category = 'General'
                for keyword, cat_name in categories.items():
                    if keyword.lower() in url.lower():
                        category = cat_name
                        break
                
                # Make request with timeout
                response = requests.get(url, headers=headers, timeout=8)
                
                # Check if results exist
                html = response.text.lower()
                
                # Indicators that results were found
                has_results = False
                result_count = 0
                
                # Check for "no results" indicators
                no_results_phrases = [
                    'did not match any documents',
                    'no results found',
                    'your search did not match',
                    '0 results',
                    'keine ergebnisse',  # German
                    'aucun résultat'     # French
                ]
                
                # If any "no results" phrase found, skip this URL
                if any(phrase in html for phrase in no_results_phrases):
                    continue
                
                # Look for result indicators
                if 'results' in html or 'search-result' in html or 'result' in html:
                    has_results = True
                    
                    # Try to estimate result count
                    count_match = re.search(r'about ([\d,]+) results?', html)
                    if count_match:
                        result_count = int(count_match.group(1).replace(',', ''))
                    else:
                        # Alternative patterns
                        count_match = re.search(r'([\d,]+) results?', html)
                        if count_match:
                            result_count = int(count_match.group(1).replace(',', ''))
                        else:
                            result_count = 1  # At least some results
                
                if has_results and result_count > 0:
                    validated.append({
                        'url': url,
                        'category': category,
                        'result_count': result_count
                    })
                
                # Rate limiting - small delay between requests
                time.sleep(0.4)
                
            except requests.RequestException:
                # Skip URLs that fail (timeout, connection error, etc.)
                continue
            except Exception:
                continue
        
        print(f"    ✓ Validation complete: {len(validated)} URLs with actual results")
        return validated
    
    def run_phoneinfoga(self, phone_number, output_dir):
        """Run PhoneInfoga phone number search with result validation"""
        print(f"\n[*] Running PhoneInfoga on: {phone_number}")
        
        # Check if PhoneInfoga is installed
        setup = self.check_phoneinfoga_setup()
        
        if setup['status'] != 'ok':
            print(f"[-] {setup['message']}")
            if 'install_cmd' in setup:
                print(f"    {setup['install_cmd']}")
            return False
        
        # Get the executable path
        phoneinfoga_cmd = setup.get('path', 'phoneinfoga')
        
        # Sanitize phone number for filename
        safe_number = self.sanitize_phone_number(phone_number)
        output_txt = output_dir / f"{safe_number}_phoneinfoga_output.txt"
        output_json = output_dir / f"{safe_number}_phoneinfoga_results.json"
        output_results = output_dir / f"{safe_number}_phoneinfoga_results.txt"
        output_validated = output_dir / f"{safe_number}_phoneinfoga_VALIDATED.txt"
        
        # Run PhoneInfoga scan
        cmd = [
            phoneinfoga_cmd, "scan",
            "-n", phone_number
        ]
        
        try:
            print(f"    Scanning phone number...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                encoding='utf-8',
                errors='replace'
            )
            
            # Save raw output
            with open(output_txt, 'w', encoding='utf-8', errors='replace') as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n\n=== STDERR ===\n")
                    f.write(result.stderr)
            
            # Parse results
            found_info = {
                'carrier': None,
                'country': None,
                'location': None,
                'line_type': None,
                'international_format': None,
                'local_format': None,
                'valid': None,
                'urls_found': [],
                'validated_urls': [],  # NEW: Only URLs with actual results
                'social_media': []
            }
            
            # Extract information from output
            for line in result.stdout.split('\n'):
                line_lower = line.lower()
                if 'country' in line_lower and ':' in line:
                    found_info['country'] = line.split(':', 1)[1].strip()
                elif 'carrier' in line_lower and ':' in line:
                    found_info['carrier'] = line.split(':', 1)[1].strip()
                elif ('location' in line_lower or 'area' in line_lower) and ':' in line:
                    found_info['location'] = line.split(':', 1)[1].strip()
                elif ('line type' in line_lower or 'type' in line_lower) and ':' in line:
                    found_info['line_type'] = line.split(':', 1)[1].strip()
                elif 'international' in line_lower and ':' in line:
                    found_info['international_format'] = line.split(':', 1)[1].strip()
                elif 'local' in line_lower and ':' in line:
                    found_info['local_format'] = line.split(':', 1)[1].strip()
                elif 'valid' in line_lower and ':' in line:
                    found_info['valid'] = line.split(':', 1)[1].strip()
            
            # Extract ALL URLs
            found_info['urls_found'] = self.extract_urls_from_text(result.stdout)
            
            # NEW: Validate URLs to find which have actual results
            print(f"\n    📊 Generated {len(found_info['urls_found'])} search URLs")
            found_info['validated_urls'] = self.validate_phone_search_urls(
                found_info['urls_found'], 
                phone_number
            )
            
            # Save parsed results with ALL URLs
            with open(output_results, 'w', encoding='utf-8') as f:
                f.write(f"PhoneInfoga Results for: {phone_number}\n")
                f.write("="*70 + "\n\n")
                
                f.write("BASIC INFORMATION:\n")
                f.write("-"*70 + "\n")
                f.write(f"Valid: {found_info['valid'] or 'Unknown'}\n")
                f.write(f"Country: {found_info['country'] or 'Unknown'}\n")
                f.write(f"Carrier: {found_info['carrier'] or 'Unknown'}\n")
                f.write(f"Location: {found_info['location'] or 'Unknown'}\n")
                f.write(f"Line Type: {found_info['line_type'] or 'Unknown'}\n")
                f.write(f"International Format: {found_info['international_format'] or 'N/A'}\n")
                f.write(f"Local Format: {found_info['local_format'] or 'N/A'}\n\n")
                
                f.write(f"SEARCH URLS GENERATED ({len(found_info['urls_found'])} total):\n")
                f.write("-"*70 + "\n")
                for url in found_info['urls_found']:
                    f.write(f"{url}\n")
            
            # Save ONLY validated URLs with results
            if found_info['validated_urls']:
                with open(output_validated, 'w', encoding='utf-8') as f:
                    f.write(f"PhoneInfoga Validated Results for: {phone_number}\n")
                    f.write("="*70 + "\n")
                    f.write(f"✓ FOUND {len(found_info['validated_urls'])} URLs WITH ACTUAL RESULTS\n")
                    f.write("="*70 + "\n\n")
                    
                    for i, item in enumerate(found_info['validated_urls'], 1):
                        f.write(f"{i}. Category: {item['category']}\n")
                        f.write(f"   Results: ~{item['result_count']} found\n")
                        f.write(f"   URL: {item['url']}\n")
                        f.write("-"*70 + "\n")
            
            # Save JSON for programmatic access
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(found_info, f, indent=2)
            
            print(f"\n[+] PhoneInfoga completed!")
            print(f"    Country: {found_info['country'] or 'Unknown'}")
            print(f"    Carrier: {found_info['carrier'] or 'Unknown'}")
            print(f"    URLs Generated: {len(found_info['urls_found'])}")
            print(f"    ✓ URLs with Results: {len(found_info['validated_urls'])}")
            
            if found_info['validated_urls']:
                print(f"\n    💡 ACTIONABLE LEADS FOUND:")
                for item in found_info['validated_urls']:
                    print(f"       • {item['category']}: ~{item['result_count']} results")
                print(f"\n    ⭐ Check: {output_validated.name}")
            else:
                print(f"    ℹ️  No public records found (good privacy!)")
            
            return True
            
        except subprocess.TimeoutExpired:
            print(f"[-] PhoneInfoga timed out after 2 minutes")
            return False
        except Exception as e:
            print(f"[-] PhoneInfoga error: {e}")
            return False
    
    def run_spiderfoot(self, target):
        """Run SpiderFoot scan"""
        print(f"\n[*] SpiderFoot: Starting web interface...")
        print(f"[!] SpiderFoot requires manual operation via web UI")
        print(f"[>] Navigate to: http://127.0.0.1:5001")
        print(f"[>] Target: {target}")
        
        sf_path = self.tools['spiderfoot']
        
        if not sf_path.exists():
            print(f"[-] SpiderFoot not found at {sf_path}")
            print(f"    Clone it: git clone https://github.com/smicallef/spiderfoot")
            return False
        
        try:
            print(f"\nPress Ctrl+C to stop SpiderFoot when done\n")
            subprocess.run([sys.executable, str(sf_path), "-l", "127.0.0.1:5001"])
        except KeyboardInterrupt:
            print(f"\n[+] SpiderFoot stopped")
        return True
    
    def search_username(self, username, tools=['maigret', 'sherlock', 'whatsmyname']):
        """Search username across multiple tools"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.results_dir / f"{username}_{timestamp}"
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"  USERNAME SEARCH: {username}")
        print(f"  Output Directory: {output_dir}")
        print(f"  Tools: {', '.join(tools)}")
        print(f"{'='*70}")
        
        results = {}
        
        if 'maigret' in tools:
            results['maigret'] = self.run_maigret(username, output_dir)
            
        if 'sherlock' in tools:
            results['sherlock'] = self.run_sherlock(username, output_dir)
            
        if 'whatsmyname' in tools:
            results['whatsmyname'] = self.run_whatsmyname(username, output_dir)
        
        # List created files
        print(f"\n{'='*70}")
        print(f"  FILES CREATED")
        print(f"{'='*70}")
        
        files = sorted(output_dir.glob('*'))
        if files:
            for f in files:
                size = f.stat().st_size
                if 'results' in f.name:
                    indicator = "⭐"
                elif f.suffix == '.html':
                    indicator = "📄"
                elif f.suffix == '.csv':
                    indicator = "📊"
                else:
                    indicator = "📝"
                print(f"  {indicator} {f.name} ({size:,} bytes)")
        else:
            print(f"  [!] No files created")
        
        # Summary
        print(f"\n{'='*70}")
        print(f"  SEARCH COMPLETE")
        print(f"{'='*70}")
        
        for tool, success in results.items():
            status = "[+]" if success else "[!]"
            print(f"  {status} {tool.capitalize()}")
        
        print(f"\nResults saved to: {output_dir}")
        
        if sys.platform == 'win32':
            try:
                os.startfile(output_dir)
            except:
                pass
        
        return output_dir
    
    def search_domain(self, domain):
        """Search domain with theHarvester"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.results_dir / f"{domain.replace('.', '_')}_{timestamp}"
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"  DOMAIN SEARCH: {domain}")
        print(f"  Output Directory: {output_dir}")
        print(f"{'='*70}")
        
        self.run_theharvester(domain, output_dir)
        
        # List files
        print(f"\n{'='*70}")
        print(f"  FILES CREATED")
        print(f"{'='*70}")
        
        files = sorted(output_dir.glob('*'))
        if files:
            for f in files:
                size = f.stat().st_size
                if 'results' in f.name:
                    indicator = "⭐"
                else:
                    indicator = "📝"
                print(f"  {indicator} {f.name} ({size:,} bytes)")
        else:
            print(f"  [!] No files created")
        
        print(f"\nResults saved to: {output_dir}")
        return output_dir
    
    def search_phone(self, phone_number):
        """Search phone number with PhoneInfoga"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_number = self.sanitize_phone_number(phone_number)
        output_dir = self.results_dir / f"phone_{safe_number}_{timestamp}"
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"  PHONE NUMBER SEARCH: {phone_number}")
        print(f"  Output Directory: {output_dir}")
        print(f"{'='*70}")
        
        self.run_phoneinfoga(phone_number, output_dir)
        
        # List files
        print(f"\n{'='*70}")
        print(f"  FILES CREATED")
        print(f"{'='*70}")
        
        files = sorted(output_dir.glob('*'))
        if files:
            for f in files:
                size = f.stat().st_size
                if 'VALIDATED' in f.name:
                    indicator = "🎯"  # Special indicator for validated results
                elif 'results' in f.name:
                    indicator = "⭐"
                elif f.suffix == '.json':
                    indicator = "📊"
                else:
                    indicator = "📝"
                print(f"  {indicator} {f.name} ({size:,} bytes)")
        else:
            print(f"  [!] No files created")
        
        print(f"\nResults saved to: {output_dir}")
        
        if sys.platform == 'win32':
            try:
                os.startfile(output_dir)
            except:
                pass
        
        return output_dir
    
    def batch_search(self, targets_file, search_type='username'):
        """Batch search from file"""
        if not Path(targets_file).exists():
            print(f"[-] File not found: {targets_file}")
            return
        
        with open(targets_file, 'r') as f:
            targets = [line.strip() for line in f if line.strip()]
        
        print(f"\n{'='*70}")
        print(f"  BATCH SEARCH MODE")
        print(f"  Targets: {len(targets)}")
        print(f"  Type: {search_type}")
        print(f"{'='*70}")
        
        results = []
        for i, target in enumerate(targets, 1):
            print(f"\n[{i}/{len(targets)}] Processing: {target}")
            
            if search_type == 'username':
                output = self.search_username(target, tools=['maigret'])
            elif search_type == 'domain':
                output = self.search_domain(target)
            elif search_type == 'phone':
                output = self.search_phone(target)
            
            results.append({'target': target, 'output': str(output)})
            
            if i < len(targets):
                wait_time = 10
                print(f"\n[*] Waiting {wait_time} seconds before next search...")
                time.sleep(wait_time)
        
        summary_file = self.results_dir / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[+] Batch search complete")
        print(f"Summary saved to: {summary_file}")
    
    def interactive_menu(self):
        """Interactive menu interface"""
        while True:
            self.print_banner()
            
            print(f"\nSelect an option:")
            print(f"  1 - Single Username Search (All 3 tools)")
            print(f"  2 - Single Username Search (Maigret only - most reliable)")
            print(f"  3 - Domain/Email Search (theHarvester)")
            print(f"  4 - Phone Number Search (PhoneInfoga + Validation)")
            print(f"  5 - Batch Search (from file)")
            print(f"  6 - SpiderFoot (Web UI)")
            print(f"  7 - Check Tool Status")
            print(f"  0 - Exit")
            
            choice = input(f"\nEnter choice: ").strip()
            
            if choice == '1':
                username = input(f"Enter username: ").strip()
                if username:
                    self.search_username(username)
                    input(f"\nPress Enter to continue...")
                
            elif choice == '2':
                username = input(f"Enter username: ").strip()
                if username:
                    self.search_username(username, tools=['maigret'])
                    input(f"\nPress Enter to continue...")
                
            elif choice == '3':
                domain = input(f"Enter domain (e.g., example.com): ").strip()
                if domain:
                    self.search_domain(domain)
                    input(f"\nPress Enter to continue...")
                
            elif choice == '4':
                phone = input(f"Enter phone number (e.g., +1234567890): ").strip()
                if phone:
                    self.search_phone(phone)
                    input(f"\nPress Enter to continue...")
                
            elif choice == '5':
                file_path = input(f"Enter path to targets file: ").strip()
                search_type = input(f"Type (username/domain/phone): ").strip().lower()
                if file_path and search_type in ['username', 'domain', 'phone']:
                    self.batch_search(file_path, search_type)
                    input(f"\nPress Enter to continue...")
                else:
                    print(f"[!] Invalid type. Must be: username, domain, or phone")
                    time.sleep(2)
                
            elif choice == '6':
                target = input(f"Enter target (username/domain/IP): ").strip()
                if target:
                    self.run_spiderfoot(target)
                    input(f"\nPress Enter to continue...")
            
            elif choice == '7':
                self.check_all_tools()
                input(f"\nPress Enter to continue...")
                
            elif choice == '0':
                print(f"\nGoodbye!\n")
                break
            
            else:
                print(f"Invalid choice. Try again.")
                time.sleep(1)
    
    def check_all_tools(self):
        """Check status of all tools"""
        print(f"\n{'='*70}")
        print(f"  TOOL STATUS CHECK")
        print(f"{'='*70}\n")
        
        # Check Maigret
        try:
            result = subprocess.run(
                [sys.executable, "-m", "maigret", "--version"], 
                capture_output=True, 
                timeout=5
            )
            if result.returncode == 0:
                print(f"[+] Maigret: OK")
            else:
                print(f"[!] Maigret: Installed but not responding")
        except:
            print(f"[!] Maigret: Not installed (pip install maigret)")
        
        # Check Sherlock
        try:
            result = subprocess.run(
                ["sherlock", "--version"], 
                capture_output=True, 
                timeout=5
            )
            if result.returncode == 0:
                print(f"[+] Sherlock: OK")
            else:
                print(f"[!] Sherlock: Installed but not responding")
        except:
            print(f"[!] Sherlock: Not installed (pip install sherlock-project)")
        
        # Check WhatsMyName
        wmn_path = self.tools['whatsmyname']
        if wmn_path.exists():
            print(f"[+] WhatsMyName: OK")
        else:
            print(f"[!] WhatsMyName: Not found at {wmn_path}")
        
        # Check theHarvester
        setup = self.check_theharvester_setup()
        if setup['status'] == 'ok':
            print(f"[+] theHarvester: OK")
        else:
            print(f"[!] theHarvester: {setup['message']}")
            if 'install_cmd' in setup:
                print(f"    Install with: {setup['install_cmd']}")
        
        # Check PhoneInfoga
        setup = self.check_phoneinfoga_setup()
        if setup['status'] == 'ok':
            print(f"[+] PhoneInfoga: OK ({setup['type']})")
        else:
            print(f"[!] PhoneInfoga: {setup['message']}")
            if 'install_cmd' in setup:
                print(f"    {setup['install_cmd']}")
        
        # Check requests library (needed for validation)
        try:
            import requests
            print(f"[+] requests library: OK (enables PhoneInfoga validation)")
        except ImportError:
            print(f"[!] requests library: Not installed")
            print(f"    Install with: pip install requests")
        
        # Check SpiderFoot
        sf_path = self.tools['spiderfoot']
        if sf_path.exists():
            print(f"[+] SpiderFoot: OK")
        else:
            print(f"[!] SpiderFoot: Not found at {sf_path}")

def main():
    parser = argparse.ArgumentParser(description='Unified OSINT Search Tool v3.6')
    parser.add_argument('-u', '--username', help='Username to search')
    parser.add_argument('-d', '--domain', help='Domain to search')
    parser.add_argument('-p', '--phone', help='Phone number to search (e.g., +1234567890)')
    parser.add_argument('-f', '--file', help='File containing targets (one per line)')
    parser.add_argument('-t', '--type', choices=['username', 'domain', 'phone'], default='username',
                        help='Type of targets in file')
    parser.add_argument('--tools', help='Comma-separated list of tools (maigret,sherlock,whatsmyname)',
                        default='maigret,sherlock,whatsmyname')
    parser.add_argument('--check', action='store_true', help='Check tool status')
    
    args = parser.parse_args()
    
    osint = OsintMaster()
    
    if args.check:
        osint.check_all_tools()
    elif args.username:
        tools = [t.strip() for t in args.tools.split(',')]
        osint.search_username(args.username, tools=tools)
    elif args.domain:
        osint.search_domain(args.domain)
    elif args.phone:
        osint.search_phone(args.phone)
    elif args.file:
        osint.batch_search(args.file, args.type)
    else:
        osint.interactive_menu()

if __name__ == "__main__":
    main()
