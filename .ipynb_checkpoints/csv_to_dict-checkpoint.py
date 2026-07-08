import csv
import json
import os

def convert_baseline(csv_path, out_json_path):
    completions = {}

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        for i, row in enumerate(reader):
            if len(row) < 2:
                continue

            behavior_id = f"B{i:04d}"
            prompt = row[0]
            generation = row[2]

            completions[behavior_id] = [{
                "test_case": prompt,
                "generation": generation
            }]

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(completions, f, indent=2)


def convert_attack(csv_path, out_json_path):
    completions = {}

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        for i, row in enumerate(reader):
            if len(row) < 3:
                continue

            behavior_id = f"B{i:04d}"
            prompt = row[0]
            prefill = row[1]
            continuation = row[2]

            # If you saved full_generation as row[3], prefer that
            if len(row) >= 4:
                generation = row[3]
            else:
                generation = (prefill + continuation).strip()

            completions[behavior_id] = [{
                "test_case": prompt,
                "generation": generation
            }]

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(completions, f, indent=2)


convert_baseline(
    "rampup/no-attack_test_out.csv",
    "harmbench_inputs/no_attack_completions.json"
)

convert_attack(
    "rampup/attack_test_out.csv",
    "harmbench_inputs/prefill_attack_completions.json"
)
def make_behaviors(csv_path, behaviors_path):
    os.makedirs(os.path.dirname(behaviors_path), exist_ok=True)

    with open(csv_path, "r", newline="", encoding="utf-8") as infile, \
         open(behaviors_path, "w", newline="", encoding="utf-8") as outfile:

        reader = csv.reader(infile)
        writer = csv.DictWriter(
            outfile,
            fieldnames=["BehaviorID", "Behavior", "ContextString", "Tags"]
        )

        writer.writeheader()

        for i, row in enumerate(reader):
            if len(row) < 1:
                continue

            prompt = row[0]

            writer.writerow({
                "BehaviorID": f"B{i:04d}",
                "Behavior": prompt,
                "ContextString": "",
                "Tags": ""
            })


make_behaviors(
    "rampup/no-attack_test_out.csv",
    "harmbench_inputs/behaviors.csv"
)
