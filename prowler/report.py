import json
import re

def parse_prowler_log(log_file):
    report = []
    
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(r'\{.*\}', line)
            if match:
                try:
                    log_entry = json.loads(match.group())
                    if "response" in log_entry:
                        report.append({
                            "file": log_entry.get("event", "Unknown file"),
                            "response": log_entry["response"]
                        })
                except json.JSONDecodeError:
                    continue
    
    return report

if __name__ == "__main__":
    log_file = "prowler.log"
    report_data = parse_prowler_log(log_file)
    
    for entry in report_data:
        print("File:", entry["file"])
        print("Response:", entry["response"])
        print("-" * 80)