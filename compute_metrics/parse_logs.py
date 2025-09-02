import re
from typing import List, Dict, Any


def _split_cases(case_str: str) -> List[str]:
    return [c.strip() for c in case_str.split(",") if c.strip()]


def parse_logs(log_text: str) -> List[Dict[str, Any]]:
    samples = re.split(r'Sample: \d+\.', log_text)[1:]
    results = []

    for i, sample in enumerate(samples, start=1):
        final_answer_match = re.search(r'FINALLY ANSWER:(.*?)(?=Sample:|$)', sample, re.DOTALL)
        final_answer = final_answer_match.group(1).strip() if final_answer_match else None

        case_match = re.search(r'Case:\s*(.+?)\n', sample)
        sample_cases = _split_cases(case_match.group(1)) if case_match else []

        tool_blocks = re.findall(
            r'TOOL:\s*(.*?)\n.*?GEN_MOLECULES:\s*(\{.*?\})\s*END_GEN_MOLECULES',
            sample,
            re.DOTALL
        )

        found_cases = {}
        for case_name, molecules_block in tool_blocks:
            case_name = case_name.strip()
            try:
                molecules_dict = eval(molecules_block.strip())
            except Exception:
                molecules_dict = None

            molecules = molecules_dict.get("Molecules", []) if molecules_dict else []

            all_molecules_in_final = all(
                mol in final_answer for mol in molecules
            ) if final_answer and molecules else False

            found_cases[case_name] = {
                "case": case_name,
                "molecules_dict": molecules_dict,
                "molecules": molecules,
                "all_molecules_in_final": all_molecules_in_final,
            }

        cases = []
        for case in sample_cases:
            if case in found_cases:
                cases.append(found_cases[case])
            else:
                cases.append({
                    "case": case,
                    "molecules_dict": None,
                    "molecules": [],
                    "all_molecules_in_final": False,
                })

        results.append({
            "sample": i,
            "cases": cases,
            "final_answer": final_answer,
        })

    return results


def print_results_table(results: List[Dict[str, Any]]):
    print("| Sample | Case | #Molecules | All Molecules In Final |")
    print("|--------|------|------------|-------------------------|")
    for res in results:
        for c in res["cases"]:
            print(f"| {res['sample']} | {c['case']} | {len(c['molecules'])} | {c['all_molecules_in_final']} |")

def compute_accuracy(results: List[Dict[str, Any]]) -> float:
    ssa = 0
    ts = 0

    for res in results:
        tool_score = 0
        summary_score = 0
        
        for c in res["cases"]:
            if c['molecules_dict'] == None:
                summary_score += 1
                continue
            else:
                if c['all_molecules_in_final']:
                    summary_score += 1
                    tool_score +=1
        if tool_score == len(res["cases"]):
            ts += 1
        if summary_score == len(res["cases"]):
            ssa += 1
                
    print(f"Total Samples: {len(results)}")
    print("TS (tool selection, %): ", ts / len(results) * 100)
    print("SSA (summary, %): ", ssa / len(results) * 100)
    print('Finally accuracy (%): ', (ts + ssa) / (2 * len(results)) * 100)

if __name__ == "__main__":
    path = 'result_2.txt'
    with open(path, 'r') as f:
        log_text = f.read()

    results = parse_logs(log_text)
    compute_accuracy(results)
